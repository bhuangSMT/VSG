#include "SweptVolume.h"

#include <algorithm>
#include <cmath>
#include <vector>

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

void addTriangle(TriangleMesh& mesh,
                 const vsg::dvec3& a, const vsg::dvec3& b, const vsg::dvec3& c)
{
    const vsg::dvec3 ab = b - a;
    const vsg::dvec3 ac = c - a;
    vsg::dvec3 n = vsg::cross(ab, ac);
    const double len = vsg::length(n);
    if (len <= 1.0e-18) return;
    n /= len;

    MeshTriangle tri;
    tri.normal = vsg::vec3(static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z));
    tri.v0 = vsg::vec3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
    tri.v1 = vsg::vec3(static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z));
    tri.v2 = vsg::vec3(static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z));
    mesh.triangles.push_back(tri);
}

void stitchQuad(TriangleMesh& mesh,
                const vsg::dvec3& a00, const vsg::dvec3& a01,
                const vsg::dvec3& a11, const vsg::dvec3& a10)
{
    addTriangle(mesh, a00, a01, a11);
    addTriangle(mesh, a00, a11, a10);
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
                  double radius, int azimuthSegments)
{
    for (int k = 0; k < azimuthSegments; ++k)
    {
        const int k1 = (k + 1) % azimuthSegments;
        stitchQuad(mesh,
                   capsuleRingPoint(a0, aSide, aAxis, radius, k, azimuthSegments),
                   capsuleRingPoint(a0, aSide, aAxis, radius, k1, azimuthSegments),
                   capsuleRingPoint(b0, bSide, bAxis, radius, k1, azimuthSegments),
                   capsuleRingPoint(b0, bSide, bAxis, radius, k, azimuthSegments));
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
    for (int k = 0; k < azimuthSegments; ++k)
    {
        const int k1 = (k + 1) % azimuthSegments;
        const vsg::dvec3 p0 = capsuleRingPoint(centre, side, axis, radius, k, azimuthSegments);
        const vsg::dvec3 p1 = capsuleRingPoint(centre, side, axis, radius, k1, azimuthSegments);
        if (flipWinding) addTriangle(mesh, centre, p1, p0);
        else addTriangle(mesh, centre, p0, p1);
    }
}

// Half-disk in plane ⊥ dir, bulging toward `toward` (unit, ⊥ dir).
void appendHalfDisk(TriangleMesh& mesh,
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

    for (int k = 0; k < segments; ++k)
    {
        const double a0 = -0.5 * pi + pi * (static_cast<double>(k) / static_cast<double>(segments));
        const double a1 = -0.5 * pi + pi * (static_cast<double>(k + 1) / static_cast<double>(segments));
        const vsg::dvec3 p0 = centre + (x * std::cos(a0) + y * std::sin(a0)) * radius;
        const vsg::dvec3 p1 = centre + (x * std::cos(a1) + y * std::sin(a1)) * radius;
        if (flipWinding) addTriangle(mesh, centre, p1, p0);
        else addTriangle(mesh, centre, p0, p1);
    }
}

// Planar stadium face (Minkowski of a disk with a segment) in planes ≈ ⊥ dir.
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
    appendHalfDisk(mesh, cA, dir, -along, radius, segments, flipWinding);
    appendHalfDisk(mesh, cB, dir, along, radius, segments, flipWinding);

    const vsg::dvec3 aPos = cA + perp * radius;
    const vsg::dvec3 aNeg = cA - perp * radius;
    const vsg::dvec3 bPos = cB + perp * radius;
    const vsg::dvec3 bNeg = cB - perp * radius;
    if (flipWinding) stitchQuad(mesh, aPos, aNeg, bNeg, bPos);
    else stitchQuad(mesh, aPos, bPos, bNeg, aNeg);
}

// Vertical half-cylinder of radius R from tip to tip+height*dir, outer hemi
// facing `outward` (⊥ dir).
void appendVerticalHemiCylinder(TriangleMesh& mesh,
                                const vsg::dvec3& tip,
                                const vsg::dvec3& dir,
                                const vsg::dvec3& outward,
                                double radius,
                                double height,
                                int segments)
{
    if (radius <= 0.0 || height <= 0.0 || segments < 2) return;
    const vsg::dvec3 top = tip + dir * height;
    const vsg::dvec3 y = normalizeOr(outward - dir * vsg::dot(outward, dir), outward);
    const vsg::dvec3 x = normalizeOr(vsg::cross(dir, y), vsg::dvec3(1.0, 0.0, 0.0));

    for (int k = 0; k < segments; ++k)
    {
        const double a0 = -0.5 * pi + pi * (static_cast<double>(k) / static_cast<double>(segments));
        const double a1 = -0.5 * pi + pi * (static_cast<double>(k + 1) / static_cast<double>(segments));
        const vsg::dvec3 r0 = (x * std::cos(a0) + y * std::sin(a0)) * radius;
        const vsg::dvec3 r1 = (x * std::cos(a1) + y * std::sin(a1)) * radius;
        stitchQuad(mesh, tip + r0, tip + r1, top + r1, top + r0);
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
                        bool includeBottom = true)
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
        if (includeBottom) appendFullDisk(mesh, bottom, side, axis, radius, segments, true);
        appendFullDisk(mesh, top, side, axis, radius, segments, false);
        return;
    }

    vsg::dvec3 delta = tipPosB - tipPosA;
    delta -= midDir * vsg::dot(delta, midDir);
    const vsg::dvec3 alongPlanar = normalizeOr(delta, along);
    const vsg::dvec3 perp = normalizeOr(vsg::cross(midDir, alongPlanar), sideA);

    if (includeBottom)
        appendStadiumFace(mesh, tipPosA, tipPosB, midDir, radius, segments, true);
    appendStadiumFace(mesh, topA, topB, midDir, radius, segments, false);

    stitchQuad(mesh,
               tipPosA + perp * radius, tipPosB + perp * radius,
               topB + perp * radius, topA + perp * radius);
    stitchQuad(mesh,
               tipPosA - perp * radius, topA - perp * radius,
               topB - perp * radius, tipPosB - perp * radius);

    appendVerticalHemiCylinder(mesh, tipPosA, dirA, -alongPlanar, radius, height, segments);
    appendVerticalHemiCylinder(mesh, tipPosB, dirB, alongPlanar, radius, height, segments);
}

void appendFlatNoseSweep(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         double height,
                         int segments)
{
    appendStadiumPrism(mesh, tipA, tipB, radius, height, segments);
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

        for (int k = 0; k < azimuthSegments; ++k)
        {
            const int k1 = (k + 1) % azimuthSegments;
            stitchQuad(mesh,
                       capsuleRingPoint(cA0, sideA, axisA, r0, k, azimuthSegments),
                       capsuleRingPoint(cA0, sideA, axisA, r0, k1, azimuthSegments),
                       capsuleRingPoint(cA1, sideA, axisA, r1, k1, azimuthSegments),
                       capsuleRingPoint(cA1, sideA, axisA, r1, k, azimuthSegments));
            stitchQuad(mesh,
                       capsuleRingPoint(cB0, sideB, axisB, r0, k, azimuthSegments),
                       capsuleRingPoint(ringCentre(tipB, dirB, z1), sideB, axisB, r1, k, azimuthSegments),
                       capsuleRingPoint(ringCentre(tipB, dirB, z1), sideB, axisB, r1, k1, azimuthSegments),
                       capsuleRingPoint(cB0, sideB, axisB, r0, k1, azimuthSegments));
            stitchQuad(mesh,
                       capsuleRingPoint(cA0, sideA, axisA, r0, k, azimuthSegments),
                       capsuleRingPoint(cA0, sideA, axisA, r0, k1, azimuthSegments),
                       capsuleRingPoint(cB0, sideB, axisB, r0, k1, azimuthSegments),
                       capsuleRingPoint(cB0, sideB, axisB, r0, k, azimuthSegments));
        }
    }
}

void appendBullNoseSweep(TriangleMesh& mesh,
                         const ToolPose& tipA,
                         const ToolPose& tipB,
                         double radius,
                         double height,
                         int segments)
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
    appendStadiumPrism(mesh, shankA, shankB, radius, height, segments, false);
}

void appendHemisphere(TriangleMesh& mesh,
                      const vsg::dvec3& centre,
                      const vsg::dvec3& outward,
                      const vsg::dvec3& side,
                      const vsg::dvec3& axis,
                      double radius,
                      int azimuthSegments,
                      int stacks)
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
                addTriangle(mesh, a[static_cast<std::size_t>(k)],
                            a[static_cast<std::size_t>(k1)], b[0]);
            }
        }
        else
        {
            for (int k = 0; k < azimuthSegments; ++k)
            {
                const int k1 = (k + 1) % azimuthSegments;
                stitchQuad(mesh,
                           a[static_cast<std::size_t>(k)], a[static_cast<std::size_t>(k1)],
                           b[static_cast<std::size_t>(k1)], b[static_cast<std::size_t>(k)]);
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
                         int azimuthSegments)
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
    appendHemisphere(mesh, cA, -along, side, axis, radius, azimuthSegments, capRevolveSteps);
    appendHemisphere(mesh, cB, along, side, axis, radius, azimuthSegments, capRevolveSteps);
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
                           int stacks)
{
    TriangleMesh temp;
    appendHemisphere(temp, centre, outward, side, axis, radius, azimuthSegments, stacks);
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
                         int azimuthSegments)
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
    appendCapsuleCylinder(mesh, cA, cB, side, axis, radius, azimuthSegments);

    vsg::dvec3 tipSideA, tipAxisA, tipSideB, tipAxisB;
    capsuleLateralFrame(dirA, tipSideA, tipAxisA);
    capsuleLateralFrame(dirB, tipSideB, tipAxisB);
    if (vsg::dot(tipSideA, tipSideB) < 0.0)
    {
        tipSideB = -tipSideB;
        tipAxisB = -tipAxisB;
    }

    const vsg::dvec3 midDir = normalizeOr(dirA + dirB, dirB);
    if (std::abs(vsg::dot(along, midDir)) > 0.92)
    {
        appendHemisphere(mesh, cA, -dirA, tipSideA, tipAxisA, radius, azimuthSegments,
                         capRevolveSteps);
        appendHemisphere(mesh, cB, -dirB, tipSideB, tipAxisB, radius, azimuthSegments,
                         capRevolveSteps);
    }
    else
    {
        appendLowerHemisphere(mesh, cA, -along, side, axis, dirA, radius, azimuthSegments,
                              capRevolveSteps);
        appendLowerHemisphere(mesh, cB, along, side, axis, dirB, radius, azimuthSegments,
                              capRevolveSteps);
    }

    if (height <= 0.0) return;

    const vsg::dvec3 topA = cA + dirA * height;
    const vsg::dvec3 topB = cB + dirB * height;

    loftRingPair(mesh, cA, tipSideA, tipAxisA, topA, tipSideA, tipAxisA, radius, azimuthSegments);
    loftRingPair(mesh, cB, tipSideB, tipAxisB, topB, tipSideB, tipAxisB, radius, azimuthSegments);
    loftRingPair(mesh, topA, tipSideA, tipAxisA, topB, tipSideB, tipAxisB, radius, azimuthSegments);
    appendTopDisc(mesh, topA, tipSideA, tipAxisA, radius, azimuthSegments, false);
    appendTopDisc(mesh, topB, tipSideB, tipAxisB, radius, azimuthSegments, true);
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

void SweptVolume::appendSegment(ToolType type,
                                float radius,
                                float height,
                                const ToolPose& tipA,
                                const ToolPose& tipB,
                                int circleSegments)
{
    if (type == ToolType::None || radius <= 0.0f) return;

    const double R = static_cast<double>(radius);
    const double H = static_cast<double>(height);

    switch (type)
    {
    case ToolType::Sphere:
        appendSphereCapsule(_mesh, tipA, tipB, R, circleSegments);
        break;
    case ToolType::BallNose:
        appendBallNoseSweep(_mesh, tipA, tipB, R, H, circleSegments);
        break;
    case ToolType::FlatNose:
        appendFlatNoseSweep(_mesh, tipA, tipB, R, H, circleSegments);
        break;
    case ToolType::BullNose:
        appendBullNoseSweep(_mesh, tipA, tipB, R, H, circleSegments);
        break;
    case ToolType::None:
        return;
    }

    rebuildBvh();
    _lastPose = tipB;
}

void SweptVolume::appendTriangles(const TriangleMesh& extra, bool rebuildHierarchy)
{
    _mesh.triangles.insert(_mesh.triangles.end(),
                           extra.triangles.begin(), extra.triangles.end());
    if (rebuildHierarchy) rebuildBvh();
}

} // namespace app
