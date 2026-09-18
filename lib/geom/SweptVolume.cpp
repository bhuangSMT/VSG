#include "SweptVolume.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "ToolGeometry.h"

namespace app
{
namespace
{

constexpr double pi = 3.14159265358979323846;

// Fixed-cost tip closure: hemisphere / fillet stacks.
constexpr int capRevolveSteps = 8;
constexpr int filletStacks = 8;

vsg::dvec3 normalizeOr(const vsg::dvec3& v, const vsg::dvec3& fallback)
{
    const double len = vsg::length(v);
    if (len <= 1.0e-12) return fallback;
    return v / len;
}

void addOrientedTriangle(TriangleMesh& mesh,
                         vsg::dvec3 a, vsg::dvec3 b, vsg::dvec3 c,
                         const vsg::dvec3& outwardHint,
                         bool motionCap = false)
{
    vsg::dvec3 n = vsg::cross(b - a, c - a);
    const double len = vsg::length(n);
    if (len <= 1.0e-18) return;
    if (vsg::dot(n, outwardHint) < 0.0)
    {
        std::swap(b, c);
        n = -n;
    }
    n /= len;

    MeshTriangle tri;
    tri.normal = vsg::vec3(static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z));
    tri.v0 = vsg::vec3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
    tri.v1 = vsg::vec3(static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z));
    tri.v2 = vsg::vec3(static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z));
    if (motionCap) tri.flags = MeshTriangle::flagMotionCap;
    mesh.triangles.push_back(tri);
}

double triangleArea(const vsg::dvec3& a, const vsg::dvec3& b, const vsg::dvec3& c)
{
    return vsg::length(vsg::cross(b - a, c - a));
}

// Ruled quad between two cap edges. Picks the diagonal that keeps both
// triangles alive, then forces the winding to face `outwardHint`.
void stitchOrientedQuad(TriangleMesh& mesh,
                        const vsg::dvec3& a00, const vsg::dvec3& a01,
                        const vsg::dvec3& a11, const vsg::dvec3& a10,
                        const vsg::dvec3& outwardHint,
                        bool motionCap = false)
{
    const double splitA = std::min(triangleArea(a00, a01, a11), triangleArea(a00, a11, a10));
    const double splitB = std::min(triangleArea(a00, a01, a10), triangleArea(a01, a11, a10));
    if (splitB > splitA)
    {
        addOrientedTriangle(mesh, a00, a01, a10, outwardHint, motionCap);
        addOrientedTriangle(mesh, a01, a11, a10, outwardHint, motionCap);
        return;
    }
    addOrientedTriangle(mesh, a00, a01, a11, outwardHint, motionCap);
    addOrientedTriangle(mesh, a00, a11, a10, outwardHint, motionCap);
}

vsg::dvec3 sphereCentre(const ToolPose& tip, double radius)
{
    return tip.position + normalizeOr(tip.direction, vsg::dvec3(0.0, 0.0, 1.0)) * radius;
}

void capsuleLateralFrame(const vsg::dvec3& along, vsg::dvec3& side, vsg::dvec3& axis)
{
    vsg::dvec3 up(0.0, 0.0, 1.0);
    if (std::abs(vsg::dot(along, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);
    side = normalizeOr(vsg::cross(up, along), vsg::dvec3(1.0, 0.0, 0.0));
    axis = vsg::cross(along, side);
}

vsg::dvec3 capsuleRingPoint(const vsg::dvec3& centre,
                            const vsg::dvec3& side,
                            const vsg::dvec3& axis,
                            double radius,
                            int k,
                            int segments)
{
    const double a = (2.0 * pi * static_cast<double>(k)) / static_cast<double>(segments);
    return centre + (side * std::cos(a) + axis * std::sin(a)) * radius;
}

void loftRingPair(TriangleMesh& mesh,
                  const vsg::dvec3& a0, const vsg::dvec3& aSide, const vsg::dvec3& aAxis,
                  const vsg::dvec3& b0, const vsg::dvec3& bSide, const vsg::dvec3& bAxis,
                  double radius, int azimuthSegments,
                  const vsg::dvec3& outwardHint = {})
{
    // Per-quad radial (quad centroid − segment mid) points out of the cylinder.
    // A single global outwardHint cannot cover every azimuth, so only use it when
    // the radial falls back (degenerate / collapsed loft).
    const vsg::dvec3 mid = (a0 + b0) * 0.5;
    const bool haveHint = vsg::length(outwardHint) > 1.0e-12;
    for (int k = 0; k < azimuthSegments; ++k)
    {
        const int k1 = (k + 1) % azimuthSegments;
        const vsg::dvec3 aK = capsuleRingPoint(a0, aSide, aAxis, radius, k, azimuthSegments);
        const vsg::dvec3 aK1 = capsuleRingPoint(a0, aSide, aAxis, radius, k1, azimuthSegments);
        const vsg::dvec3 bK1 = capsuleRingPoint(b0, bSide, bAxis, radius, k1, azimuthSegments);
        const vsg::dvec3 bK = capsuleRingPoint(b0, bSide, bAxis, radius, k, azimuthSegments);
        const vsg::dvec3 radial = (aK + aK1 + bK + bK1) * 0.25 - mid;
        const vsg::dvec3 hint =
            (vsg::length(radial) > 1.0e-12) ? radial
            : (haveHint ? outwardHint : (aK - a0));
        stitchOrientedQuad(mesh, aK, aK1, bK1, bK, hint);
    }
}

void appendFullDisk(TriangleMesh& mesh,
                    const vsg::dvec3& centre,
                    const vsg::dvec3& side,
                    const vsg::dvec3& axis,
                    double radius,
                    int azimuthSegments,
                    bool flipWinding)
{
    const vsg::dvec3 hint = flipWinding ? -vsg::cross(side, axis) : vsg::cross(side, axis);
    for (int k = 0; k < azimuthSegments; ++k)
    {
        const int k1 = (k + 1) % azimuthSegments;
        const vsg::dvec3 p0 = capsuleRingPoint(centre, side, axis, radius, k, azimuthSegments);
        const vsg::dvec3 p1 = capsuleRingPoint(centre, side, axis, radius, k1, azimuthSegments);
        addOrientedTriangle(mesh, centre, p0, p1, hint);
    }
}

// Semi-revolve of the circular top profile 180° in the plane ⊥ dir.
// Endpoints sit at ±cross(dir, toward); the bulge points along `toward`.
void semiRevolveProfile(TriangleMesh& mesh,
                        const vsg::dvec3& centre,
                        const vsg::dvec3& dir,
                        const vsg::dvec3& toward,
                        double radius,
                        int segments,
                        bool flipWinding)
{
    if (radius <= 0.0 || segments < 2) return;
    const vsg::dvec3 y = normalizeOr(toward - dir * vsg::dot(toward, dir), toward);
    const vsg::dvec3 x = normalizeOr(vsg::cross(dir, y), vsg::dvec3(1.0, 0.0, 0.0));
    const vsg::dvec3 hint = flipWinding ? -dir : dir;

    for (int k = 0; k < segments; ++k)
    {
        const double a0 = pi * (static_cast<double>(k) / static_cast<double>(segments));
        const double a1 = pi * (static_cast<double>(k + 1) / static_cast<double>(segments));
        const vsg::dvec3 p0 = centre + (x * std::cos(a0) + y * std::sin(a0)) * radius;
        const vsg::dvec3 p1 = centre + (x * std::cos(a1) + y * std::sin(a1)) * radius;
        addOrientedTriangle(mesh, centre, p0, p1, hint);
    }
}

void appendHalfDisk(TriangleMesh& mesh,
                    const vsg::dvec3& centre,
                    const vsg::dvec3& dir,
                    const vsg::dvec3& toward,
                    double radius,
                    int segments,
                    bool flipWinding)
{
    semiRevolveProfile(mesh, centre, dir, toward, radius, segments, flipWinding);
}

// Ruled strip between the two semi-revolve rims (the stadium rectangle).
void stitchRevolveBoundaries(TriangleMesh& mesh,
                             const vsg::dvec3& cA,
                             const vsg::dvec3& cB,
                             const vsg::dvec3& perp,
                             double radius,
                             const vsg::dvec3& outwardHint)
{
    const vsg::dvec3 aPos = cA + perp * radius;
    const vsg::dvec3 aNeg = cA - perp * radius;
    const vsg::dvec3 bPos = cB + perp * radius;
    const vsg::dvec3 bNeg = cB - perp * radius;
    stitchOrientedQuad(mesh, aPos, bPos, bNeg, aNeg, outwardHint);
}

// One lid: semi-revolve at A, semi-revolve at B, stitch the open rims.
void appendStadiumFace(TriangleMesh& mesh,
                       const vsg::dvec3& cA,
                       const vsg::dvec3& cB,
                       const vsg::dvec3& dir,
                       double radius,
                       int segments,
                       bool flipWinding)
{
    if (radius <= 0.0) return;

    vsg::dvec3 delta = cB - cA;
    delta -= dir * vsg::dot(delta, dir);
    const double span = vsg::length(delta);
    if (span <= 1.0e-12)
    {
        vsg::dvec3 side, axis;
        capsuleLateralFrame(dir, side, axis);
        appendFullDisk(mesh, cA, side, axis, radius, std::max(segments, 3), flipWinding);
        return;
    }

    const vsg::dvec3 along = delta / span;
    const vsg::dvec3 perp = normalizeOr(vsg::cross(dir, along), vsg::dvec3(1.0, 0.0, 0.0));
    const vsg::dvec3 hint = flipWinding ? -dir : dir;
    semiRevolveProfile(mesh, cA, dir, -along, radius, segments, flipWinding);
    semiRevolveProfile(mesh, cB, dir, along, radius, segments, flipWinding);
    stitchRevolveBoundaries(mesh, cA, cB, perp, radius, hint);
}

// Vertical half-cylinder of radius R from tip to tip+height*dir, outer hemi
// facing `outward` (⊥ dir).
void appendVerticalHemiCylinder(TriangleMesh& mesh,
                                const vsg::dvec3& tip,
                                const vsg::dvec3& dir,
                                const vsg::dvec3& outward,
                                double radius,
                                double height,
                                int segments,
                                bool motionCap = true)
{
    if (radius <= 0.0 || height <= 0.0 || segments < 2) return;
    const vsg::dvec3 top = tip + dir * height;
    const vsg::dvec3 y = normalizeOr(outward - dir * vsg::dot(outward, dir), outward);
    const vsg::dvec3 x = normalizeOr(vsg::cross(dir, y), vsg::dvec3(1.0, 0.0, 0.0));

    for (int k = 0; k < segments; ++k)
    {
        const double a0 = pi * (static_cast<double>(k) / static_cast<double>(segments));
        const double a1 = pi * (static_cast<double>(k + 1) / static_cast<double>(segments));
        const vsg::dvec3 r0 = (x * std::cos(a0) + y * std::sin(a0)) * radius;
        const vsg::dvec3 r1 = (x * std::cos(a1) + y * std::sin(a1)) * radius;
        // Out of the hemi-cylinder: average of the two ring radii (in the ⊥dir plane).
        const vsg::dvec3 out = normalizeOr(r0 + r1, y);
        stitchOrientedQuad(mesh, tip + r0, tip + r1, top + r1, top + r0, out, motionCap);
    }
}

// Minkowski of a right cylinder (radius R, height H) with tipA→tipB: stadium prism.
// includeBottom is false for a bull-nose shank sitting on the fillet (avoids an
// internal face that breaks ray/boolean pairing).
void appendStadiumPrism(TriangleMesh& mesh,
                        const ToolPose& tipA,
                        const ToolPose& tipB,
                        double radius,
                        double height,
                        int segments,
                        bool includeBottom = true,
                        bool includeStartCap = true,
                        bool includeEndCap = true)
{
    if (radius <= 0.0 || height <= 0.0) return;
    if (segments < 3) segments = 3;

    const vsg::dvec3 dirA = normalizeOr(tipA.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 dirB = normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 midDir = normalizeOr(dirA + dirB, dirB);
    const vsg::dvec3 tipPosA = tipA.position;
    const vsg::dvec3 tipPosB = tipB.position;
    const vsg::dvec3 topA = tipPosA + dirA * height;
    const vsg::dvec3 topB = tipPosB + dirB * height;
    const vsg::dvec3 along = normalizeOr(tipPosB - tipPosA, midDir);

    vsg::dvec3 sideA, axisA, sideB, axisB;
    capsuleLateralFrame(dirA, sideA, axisA);
    capsuleLateralFrame(dirB, sideB, axisB);
    if (vsg::dot(sideA, sideB) < 0.0)
    {
        sideB = -sideB;
        axisB = -axisB;
    }

    if (std::abs(vsg::dot(along, midDir)) > 0.92)
    {
        const bool forward = vsg::dot(tipPosB - tipPosA, midDir) >= 0.0;
        const vsg::dvec3 bottom = forward ? tipPosA : tipPosB;
        const vsg::dvec3 top = forward ? topB : topA;
        const vsg::dvec3 dir = forward ? dirA : dirB;
        vsg::dvec3 side, axis;
        capsuleLateralFrame(dir, side, axis);
        loftRingPair(mesh, bottom, side, axis, top, side, axis, radius, segments);
        const bool diskAtBottom = forward ? includeStartCap : includeEndCap;
        const bool diskAtTop = forward ? includeEndCap : includeStartCap;
        if (includeBottom && diskAtBottom) appendFullDisk(mesh, bottom, side, axis, radius, segments, true);
        if (diskAtTop) appendFullDisk(mesh, top, side, axis, radius, segments, false);
        return;
    }

    vsg::dvec3 delta = tipPosB - tipPosA;
    delta -= midDir * vsg::dot(delta, midDir);
    const vsg::dvec3 alongPlanar = normalizeOr(delta, along);
    const vsg::dvec3 perp = normalizeOr(vsg::cross(midDir, alongPlanar), sideA);

    if (includeBottom)
        appendStadiumFace(mesh, tipPosA, tipPosB, midDir, radius, segments, true);
    appendStadiumFace(mesh, topA, topB, midDir, radius, segments, false);

    stitchOrientedQuad(mesh,
                       tipPosA + perp * radius, tipPosB + perp * radius,
                       topB + perp * radius, topA + perp * radius, perp);
    stitchOrientedQuad(mesh,
                       tipPosA - perp * radius, topA - perp * radius,
                       topB - perp * radius, tipPosB - perp * radius, -perp);

    if (includeStartCap)
        appendVerticalHemiCylinder(mesh, tipPosA, dirA, -alongPlanar, radius, height, segments);
    if (includeEndCap)
        appendVerticalHemiCylinder(mesh, tipPosB, dirB, alongPlanar, radius, height, segments);
}

void appendFlatNoseSweep(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         double height,
                         int segments,
                         bool includeStartCap = true,
                         bool includeEndCap = true)
{
    appendStadiumPrism(mesh, tipA, tipB, radius, height, segments, true, includeStartCap,
                       includeEndCap);
}

// Quarter-torus fillet loft between flat tip rim and outer shank rim.
void appendBullFilletBand(TriangleMesh& mesh,
                          const ToolPose& tipA,
                          const ToolPose& tipB,
                          double flatR,
                          double fillet,
                          int azimuthSegments,
                          int stacks)
{
    if (fillet <= 0.0 || stacks < 1) return;

    const vsg::dvec3 dirA = normalizeOr(tipA.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 dirB = normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 midDir = normalizeOr(dirA + dirB, dirB);
    vsg::dvec3 sideA, axisA, sideB, axisB;
    capsuleLateralFrame(dirA, sideA, axisA);
    capsuleLateralFrame(dirB, sideB, axisB);
    if (vsg::dot(sideA, sideB) < 0.0)
    {
        sideB = -sideB;
        axisB = -axisB;
    }

    auto ringCentre = [](const ToolPose& tip, const vsg::dvec3& dir, double z) {
        return tip.position + dir * z;
    };

    vsg::dvec3 delta = tipB.position - tipA.position;
    const vsg::dvec3 along3 = normalizeOr(delta, midDir);
    const bool plunge = std::abs(vsg::dot(along3, midDir)) > 0.92;
    delta -= midDir * vsg::dot(delta, midDir);
    const vsg::dvec3 along = normalizeOr(delta, along3);
    const vsg::dvec3 perp = normalizeOr(vsg::cross(midDir, along), sideA);

    for (int s = 0; s < stacks; ++s)
    {
        const double t0 = static_cast<double>(s) / static_cast<double>(stacks);
        const double t1 = static_cast<double>(s + 1) / static_cast<double>(stacks);
        const double phi0 = 0.5 * pi * t0;
        const double phi1 = 0.5 * pi * t1;
        const double z0 = fillet * (1.0 - std::cos(phi0));
        const double z1 = fillet * (1.0 - std::cos(phi1));
        const double r0 = flatR + fillet * std::sin(phi0);
        const double r1 = flatR + fillet * std::sin(phi1);

        const vsg::dvec3 cA0 = ringCentre(tipA, dirA, z0);
        const vsg::dvec3 cA1 = ringCentre(tipA, dirA, z1);
        const vsg::dvec3 cB0 = ringCentre(tipB, dirB, z0);
        const vsg::dvec3 cB1 = ringCentre(tipB, dirB, z1);

        if (plunge)
        {
            for (int k = 0; k < azimuthSegments; ++k)
            {
                const int k1 = (k + 1) % azimuthSegments;
                const vsg::dvec3 a0k = capsuleRingPoint(cA0, sideA, axisA, r0, k, azimuthSegments);
                const vsg::dvec3 a0k1 = capsuleRingPoint(cA0, sideA, axisA, r0, k1, azimuthSegments);
                const vsg::dvec3 a1k = capsuleRingPoint(cA1, sideA, axisA, r1, k, azimuthSegments);
                const vsg::dvec3 a1k1 = capsuleRingPoint(cA1, sideA, axisA, r1, k1, azimuthSegments);
                const vsg::dvec3 b0k = capsuleRingPoint(cB0, sideB, axisB, r0, k, azimuthSegments);
                const vsg::dvec3 b0k1 = capsuleRingPoint(cB0, sideB, axisB, r0, k1, azimuthSegments);
                const vsg::dvec3 b1k = capsuleRingPoint(cB1, sideB, axisB, r1, k, azimuthSegments);
                const vsg::dvec3 b1k1 = capsuleRingPoint(cB1, sideB, axisB, r1, k1, azimuthSegments);
                const vsg::dvec3 outA = (a0k - cA0) * std::sin(phi0) - dirA * std::cos(phi0);
                const vsg::dvec3 outB = (b0k - cB0) * std::sin(phi0) - dirB * std::cos(phi0);
                stitchOrientedQuad(mesh, a0k, a0k1, a1k1, a1k, outA);
                stitchOrientedQuad(mesh, b0k, b1k, b1k1, b0k1, outB);
            }
            continue;
        }

        // Semi-revolve of the fillet profile at A (away from the move) and at B
        // (into the move). Endpoints sit on ±perp so the two open rims match.
        auto filletPoint = [](const vsg::dvec3& centre, const vsg::dvec3& x,
                              const vsg::dvec3& y, double radius, double theta) {
            return centre + (x * std::cos(theta) + y * std::sin(theta)) * radius;
        };

        for (int k = 0; k < azimuthSegments; ++k)
        {
            const double th0 = pi * (static_cast<double>(k) / static_cast<double>(azimuthSegments));
            const double th1 = pi * (static_cast<double>(k + 1) / static_cast<double>(azimuthSegments));
            const vsg::dvec3 a00 = filletPoint(cA0, perp, -along, r0, th0);
            const vsg::dvec3 a01 = filletPoint(cA0, perp, -along, r0, th1);
            const vsg::dvec3 a10 = filletPoint(cA1, perp, -along, r1, th0);
            const vsg::dvec3 a11 = filletPoint(cA1, perp, -along, r1, th1);
            const vsg::dvec3 b00 = filletPoint(cB0, perp, along, r0, th0);
            const vsg::dvec3 b01 = filletPoint(cB0, perp, along, r0, th1);
            const vsg::dvec3 b10 = filletPoint(cB1, perp, along, r1, th0);
            const vsg::dvec3 b11 = filletPoint(cB1, perp, along, r1, th1);
            const vsg::dvec3 outA = (a00 - cA0) * std::sin(phi0) - dirA * std::cos(phi0);
            const vsg::dvec3 outB = (b00 - cB0) * std::sin(phi0) - dirB * std::cos(phi0);
            stitchOrientedQuad(mesh, a00, a01, a11, a10, outA);
            stitchOrientedQuad(mesh, b00, b10, b11, b01, outB);
        }

        // Outer generators only — not a full-ring A–B loft (that would be another lid).
        stitchOrientedQuad(mesh,
                           cA0 + perp * r0, cA1 + perp * r1,
                           cB1 + perp * r1, cB0 + perp * r0, perp);
        stitchOrientedQuad(mesh,
                           cA0 - perp * r0, cB0 - perp * r0,
                           cB1 - perp * r1, cA1 - perp * r1, -perp);
    }
}

void appendBullNoseSweep(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         double height,
                         int segments,
                         bool includeStartCap = true,
                         bool includeEndCap = true)
{
    if (radius <= 0.0) return;
    if (segments < 3) segments = 3;

    const double fillet = std::min(radius * 0.35, radius * 0.999);
    const double flatR = radius - fillet;

    // Flat tip (disk sweep).
    if (flatR > 1.0e-9)
    {
        const vsg::dvec3 midDir = normalizeOr(
            normalizeOr(tipA.direction, vsg::dvec3(0.0, 0.0, 1.0)) +
                normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0)),
            vsg::dvec3(0.0, 0.0, 1.0));
        appendStadiumFace(mesh, tipA.position, tipB.position, midDir, flatR, segments, true);
    }

    // Corner fillet (quarter-torus loft).
    appendBullFilletBand(mesh, tipA, tipB, flatR, fillet, segments, filletStacks);

    // Shank: stadium prism of full radius starting at the fillet equator.
    ToolPose shankA = tipA;
    ToolPose shankB = tipB;
    const vsg::dvec3 dirA = normalizeOr(tipA.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 dirB = normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0));
    shankA.position = tipA.position + dirA * fillet;
    shankB.position = tipB.position + dirB * fillet;
    appendStadiumPrism(mesh, shankA, shankB, radius, height, segments, false, includeStartCap,
                       includeEndCap);
}

void appendHemisphere(TriangleMesh& mesh,
                      const vsg::dvec3& centre,
                      const vsg::dvec3& outward,
                      const vsg::dvec3& side,
                      const vsg::dvec3& axis,
                      double radius,
                      int azimuthSegments,
                      int stacks,
                      bool motionCap = false)
{
    if (radius <= 0.0 || azimuthSegments < 3 || stacks < 1) return;

    std::vector<std::vector<vsg::dvec3>> rings(static_cast<std::size_t>(stacks + 1));
    for (int s = 0; s <= stacks; ++s)
    {
        const double theta =
            (0.5 * pi) * (static_cast<double>(s) / static_cast<double>(stacks));
        const double ringR = radius * std::cos(theta);
        const double along = radius * std::sin(theta);
        auto& ring = rings[static_cast<std::size_t>(s)];

        if (s == stacks)
        {
            ring.push_back(centre + outward * radius);
            continue;
        }

        ring.reserve(static_cast<std::size_t>(azimuthSegments));
        for (int k = 0; k < azimuthSegments; ++k)
        {
            const double a =
                (2.0 * pi * static_cast<double>(k)) / static_cast<double>(azimuthSegments);
            ring.push_back(centre + outward * along +
                           (side * std::cos(a) + axis * std::sin(a)) * ringR);
        }
    }

    for (int s = 0; s < stacks; ++s)
    {
        const auto& a = rings[static_cast<std::size_t>(s)];
        const auto& b = rings[static_cast<std::size_t>(s + 1)];
        if (b.size() == 1)
        {
            for (int k = 0; k < azimuthSegments; ++k)
            {
                const int k1 = (k + 1) % azimuthSegments;
                addOrientedTriangle(mesh, a[static_cast<std::size_t>(k)],
                                    a[static_cast<std::size_t>(k1)], b[0],
                                    a[static_cast<std::size_t>(k)] - centre, motionCap);
            }
        }
        else
        {
            for (int k = 0; k < azimuthSegments; ++k)
            {
                const int k1 = (k + 1) % azimuthSegments;
                stitchOrientedQuad(mesh,
                                   a[static_cast<std::size_t>(k)], a[static_cast<std::size_t>(k1)],
                                   b[static_cast<std::size_t>(k1)], b[static_cast<std::size_t>(k)],
                                   a[static_cast<std::size_t>(k)] - centre, motionCap);
            }
        }
    }
}

void appendCapsuleCylinder(TriangleMesh& mesh,
                           const vsg::dvec3& cA,
                           const vsg::dvec3& cB,
                           const vsg::dvec3& side,
                           const vsg::dvec3& axis,
                           double radius,
                           int azimuthSegments)
{
    if (vsg::length(cB - cA) <= 1.0e-12) return;
    loftRingPair(mesh, cA, side, axis, cB, side, axis, radius, azimuthSegments);
}

void appendSphereCapsule(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         int azimuthSegments,
                         bool includeStartCap = true,
                         bool includeEndCap = true)
{
    if (radius <= 0.0) return;
    if (azimuthSegments < 3) azimuthSegments = 3;

    const vsg::dvec3 cA = sphereCentre(tipA, radius);
    const vsg::dvec3 cB = sphereCentre(tipB, radius);
    const vsg::dvec3 along =
        normalizeOr(cB - cA, normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0)));

    vsg::dvec3 side, axis;
    capsuleLateralFrame(along, side, axis);

    appendCapsuleCylinder(mesh, cA, cB, side, axis, radius, azimuthSegments);
    if (includeStartCap)
        appendHemisphere(mesh, cA, -along, side, axis, radius, azimuthSegments, capRevolveSteps, true);
    if (includeEndCap)
        appendHemisphere(mesh, cB, along, side, axis, radius, azimuthSegments, capRevolveSteps, true);
}

void appendTopDisc(TriangleMesh& mesh,
                   const vsg::dvec3& centre,
                   const vsg::dvec3& side,
                   const vsg::dvec3& axis,
                   double radius,
                   int azimuthSegments,
                   bool flipWinding)
{
    appendFullDisk(mesh, centre, side, axis, radius, azimuthSegments, flipWinding);
}

void appendLowerHemisphere(TriangleMesh& mesh,
                           const vsg::dvec3& centre,
                           const vsg::dvec3& outward,
                           const vsg::dvec3& side,
                           const vsg::dvec3& axis,
                           const vsg::dvec3& toolDir,
                           double radius,
                           int azimuthSegments,
                           int stacks,
                           bool motionCap = false)
{
    TriangleMesh temp;
    appendHemisphere(temp, centre, outward, side, axis, radius, azimuthSegments, stacks, motionCap);
    for (const MeshTriangle& tri : temp.triangles)
    {
        const vsg::dvec3 c(
            (static_cast<double>(tri.v0.x) + tri.v1.x + tri.v2.x) / 3.0,
            (static_cast<double>(tri.v0.y) + tri.v1.y + tri.v2.y) / 3.0,
            (static_cast<double>(tri.v0.z) + tri.v1.z + tri.v2.z) / 3.0);
        if (vsg::dot(c - centre, toolDir) <= 1.0e-8)
            mesh.triangles.push_back(tri);
    }
}

void appendBallNoseSweep(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         double height,
                         int azimuthSegments,
                         bool includeStartCap = true,
                         bool includeEndCap = true)
{
    if (radius <= 0.0) return;
    if (azimuthSegments < 3) azimuthSegments = 3;

    const vsg::dvec3 dirA = normalizeOr(tipA.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 dirB = normalizeOr(tipB.direction, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 cA = sphereCentre(tipA, radius);
    const vsg::dvec3 cB = sphereCentre(tipB, radius);
    const vsg::dvec3 along = normalizeOr(cB - cA, dirB);

    vsg::dvec3 side, axis;
    capsuleLateralFrame(along, side, axis);

    vsg::dvec3 tipSideA, tipAxisA, tipSideB, tipAxisB;
    capsuleLateralFrame(dirA, tipSideA, tipAxisA);
    capsuleLateralFrame(dirB, tipSideB, tipAxisB);
    if (vsg::dot(tipSideA, tipSideB) < 0.0)
    {
        tipSideB = -tipSideB;
        tipAxisB = -tipAxisB;
    }

    const vsg::dvec3 midDir = normalizeOr(dirA + dirB, dirB);
    const double span = vsg::length(cB - cA);
    if (std::abs(vsg::dot(along, midDir)) > 0.92)
    {
        appendCapsuleCylinder(mesh, cA, cB, side, axis, radius, azimuthSegments);
        if (includeStartCap)
            appendHemisphere(mesh, cA, -dirA, tipSideA, tipAxisA, radius, azimuthSegments,
                             capRevolveSteps, true);
        if (includeEndCap)
            appendHemisphere(mesh, cB, -dirB, tipSideB, tipAxisB, radius, azimuthSegments,
                             capRevolveSteps, true);
    }
    else
    {
        // Lower half of the motion capsule only. The upper half is inside the
        // shank and is a second Z-lid at the ball equator.
        appendVerticalHemiCylinder(mesh, cA, along, -midDir, radius, span, azimuthSegments, false);
        if (includeStartCap)
            appendLowerHemisphere(mesh, cA, -along, side, axis, dirA, radius, azimuthSegments,
                                  capRevolveSteps, true);
        if (includeEndCap)
            appendLowerHemisphere(mesh, cB, along, side, axis, dirB, radius, azimuthSegments,
                                  capRevolveSteps, true);
    }

    if (height <= 0.0) return;

    const vsg::dvec3 topA = cA + dirA * height;
    const vsg::dvec3 topB = cB + dirB * height;

    if (includeStartCap)
        loftRingPair(mesh, cA, tipSideA, tipAxisA, topA, tipSideA, tipAxisA, radius, azimuthSegments);
    if (includeEndCap)
        loftRingPair(mesh, cB, tipSideB, tipAxisB, topB, tipSideB, tipAxisB, radius, azimuthSegments);
    appendStadiumFace(mesh, topA, topB, midDir, radius, azimuthSegments, false);
}

} // namespace

void SweptVolume::clearGeometry()
{
    _mesh = TriangleMesh{};
    _bvh = BVH{};
}

void SweptVolume::clear()
{
    clearGeometry();
    _lastPose.reset();
}

void SweptVolume::rebuildBvh()
{
    _bvh = BVH::build(_mesh);
}

void appendMeshAtPose(TriangleMesh& dest, const TriangleMesh& local, const ToolPose& pose)
{
    const vsg::dvec3 z = normalizeOr(pose.direction, vsg::dvec3(0.0, 0.0, 1.0));
    vsg::dvec3 x, y;
    capsuleLateralFrame(z, x, y);
    // Local +X,+Y,+Z → world x,y,z frame with origin at pose.position.
    for (const MeshTriangle& tri : local.triangles)
    {
        auto xform = [&](const vsg::vec3& p) {
            return pose.position + x * static_cast<double>(p.x) + y * static_cast<double>(p.y) +
                   z * static_cast<double>(p.z);
        };
        auto xformN = [&](const vsg::vec3& n) {
            const vsg::dvec3 wn =
                x * static_cast<double>(n.x) + y * static_cast<double>(n.y) + z * static_cast<double>(n.z);
            return normalizeOr(wn, z);
        };
        const vsg::dvec3 a = xform(tri.v0);
        const vsg::dvec3 b = xform(tri.v1);
        const vsg::dvec3 c = xform(tri.v2);
        const vsg::dvec3 n = xformN(tri.normal);
        addOrientedTriangle(dest, a, b, c, n, tri.motionCap());
    }
}

vsg::dvec3 grindingWheelXform(const vsg::dvec3& local, const vsg::dvec3& origin,
                              const vsg::dvec3& x, const vsg::dvec3& y, const vsg::dvec3& z)
{
    return origin + x * local.x + y * local.y + z * local.z;
}

// One mill frame per pose (analytic feed, else neighbour chord). Dual
// incoming/outgoing stations at the same tip were what twisted the 6-gon
// when short helix steps made those chords disagree.
void appendGrindingWheelPath(TriangleMesh& mesh,
                             const std::vector<ToolPose>& poses,
                             const GrindingWheelProfile& wheel)
{
    if (poses.size() < 2) return;

    vsg::dvec3 local[6];
    const int n = grindingWheelLocalLoop(wheel, local);
    if (n < 4) return;

    struct Station
    {
        int n = 0;
        vsg::dvec3 p[6];
        vsg::dvec3 y{0.0, 1.0, 0.0};
    };

    auto stationAlong = [&](std::size_t i) {
        if (vsg::length(poses[i].feed) > 1.0e-12) return poses[i].feed;
        if (i + 1 < poses.size()) return poses[i + 1].position - poses[i].position;
        return poses[i].position - poses[i - 1].position;
    };

    auto place = [&](const ToolPose& pose, const vsg::dvec3& along,
                     const vsg::dvec3* prevY) {
        vsg::dvec3 x, y, z;
        grindingWheelMotionFrame(pose.direction, along, x, y, z, prevY);
        Station s;
        s.n = n;
        s.y = y;
        for (int i = 0; i < n; ++i)
            s.p[i] = grindingWheelXform(local[i], pose.position, x, y, z);
        return s;
    };

    std::vector<Station> stations;
    stations.reserve(poses.size());
    const vsg::dvec3* prevY = nullptr;
    vsg::dvec3 prevYStorage(0.0, 1.0, 0.0);
    for (std::size_t i = 0; i < poses.size(); ++i)
    {
        const Station s = place(poses[i], stationAlong(i), prevY);
        stations.push_back(s);
        prevYStorage = s.y;
        prevY = &prevYStorage;
    }

    const vsg::dvec3 alongStart =
        normalizeOr(poses[1].position - poses[0].position, vsg::dvec3(0.0, 0.0, 1.0));
    const vsg::dvec3 alongEnd = normalizeOr(
        poses.back().position - poses[poses.size() - 2].position, alongStart);

    auto centroid = [](const Station& s) {
        vsg::dvec3 c(0.0, 0.0, 0.0);
        for (int i = 0; i < s.n; ++i) c += s.p[i];
        return c / static_cast<double>(s.n);
    };

    auto fanCap = [&](const Station& s, const vsg::dvec3& outward) {
        for (int i = 1; i + 1 < s.n; ++i)
            addOrientedTriangle(mesh, s.p[0], s.p[i], s.p[i + 1], outward, true);
    };

    {
        const vsg::dvec3 c0 = centroid(stations.front());
        const vsg::dvec3 c1 = centroid(stations[1]);
        vsg::dvec3 startOut = c0 - c1;
        if (vsg::length(startOut) <= 1.0e-12) startOut = -alongStart;
        fanCap(stations.front(), startOut);

        const vsg::dvec3 cN = centroid(stations.back());
        const vsg::dvec3 cPrev = centroid(stations[stations.size() - 2]);
        vsg::dvec3 endOut = cN - cPrev;
        if (vsg::length(endOut) <= 1.0e-12) endOut = alongEnd;
        fanCap(stations.back(), endOut);
    }

    for (std::size_t i = 0; i + 1 < stations.size(); ++i)
    {
        const Station& a = stations[i];
        const Station& b = stations[i + 1];
        vsg::dvec3 localC(0.0, 0.0, 0.0);
        for (int k = 0; k < a.n; ++k) localC += a.p[k] + b.p[k];
        localC /= static_cast<double>(a.n * 2);

        auto sideOut = [&](const vsg::dvec3& p0, const vsg::dvec3& p1, const vsg::dvec3& q0,
                           const vsg::dvec3& q1) {
            const vsg::dvec3 edgeMid = (p0 + p1 + q0 + q1) * 0.25;
            vsg::dvec3 n = vsg::cross(p1 - p0, q0 - p0) + vsg::cross(q1 - p0, q0 - p0);
            if (vsg::dot(n, edgeMid - localC) < 0.0) n = -n;
            return n;
        };

        for (int e = 0; e < a.n; ++e)
        {
            const int e1 = (e + 1) % a.n;
            stitchOrientedQuad(mesh, a.p[e], a.p[e1], b.p[e1], b.p[e],
                               sideOut(a.p[e], a.p[e1], b.p[e], b.p[e1]));
        }
    }
}

void SweptVolume::appendSegment(ToolType type,
                                float radius,
                                float height,
                                const ToolPose& tipA,
                                const ToolPose& tipB,
                                int circleSegments,
                                float vertexAngleDeg,
                                float shankRadius,
                                float shankLength,
                                GrindingWheelProfile wheel)
{
    appendPath(type, radius, height, std::vector<ToolPose>{tipA, tipB}, circleSegments,
               vertexAngleDeg, shankRadius, shankLength, wheel);
}

void SweptVolume::appendPath(ToolType type,
                             float radius,
                             float height,
                             const std::vector<ToolPose>& poses,
                             int circleSegments,
                             float vertexAngleDeg,
                             float shankRadius,
                             float shankLength,
                             GrindingWheelProfile wheel)
{
    (void)vertexAngleDeg;
    (void)shankRadius;
    (void)shankLength;
    if (type == ToolType::None || poses.size() < 2) return;
    if (type == ToolType::GrindingWheel)
    {
        if (!wheel.valid()) return;
    }
    else if (radius <= 0.0f)
        return;

    const double R = static_cast<double>(radius);
    const double H = static_cast<double>(height);
    const std::size_t triBefore = _mesh.triangles.size();

    if (type == ToolType::GrindingWheel)
    {
        appendGrindingWheelPath(_mesh, poses, wheel);
    }
    else
    {
        for (std::size_t i = 0; i + 1 < poses.size(); ++i)
        {
            const ToolPose& tipA = poses[i];
            const ToolPose& tipB = poses[i + 1];
            const bool startCap = (i == 0);
            const bool endCap = (i + 2 == poses.size());
            switch (type)
            {
            case ToolType::Sphere:
                appendSphereCapsule(_mesh, tipA, tipB, R, circleSegments, startCap, endCap);
                break;
            case ToolType::BallNose:
                appendBallNoseSweep(_mesh, tipA, tipB, R, H, circleSegments, startCap, endCap);
                break;
            case ToolType::FlatNose:
                appendFlatNoseSweep(_mesh, tipA, tipB, R, H, circleSegments, startCap, endCap);
                break;
            case ToolType::BullNose:
                appendBullNoseSweep(_mesh, tipA, tipB, R, H, circleSegments, startCap, endCap);
                break;
            case ToolType::GrindingWheel:
            case ToolType::None:
                break;
            }
        }
    }

    if (_mesh.triangles.size() == triBefore) return;
    rebuildBvh();
    _lastPose = poses.back();
}

void SweptVolume::appendTriangles(const TriangleMesh& extra, bool rebuildHierarchy)
{
    _mesh.triangles.insert(_mesh.triangles.end(),
                           extra.triangles.begin(), extra.triangles.end());
    if (rebuildHierarchy) rebuildBvh();
}

} // namespace app
