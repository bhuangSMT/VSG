#include "SweptVolume.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace app
{
namespace
{

constexpr double pi = 3.14159265358979323846;

// Fixed-cost tip closure: 180° of profile revolution in this many steps.
constexpr int capRevolveSteps = 8;

struct ProfilePoint
{
    double r = 0.0; // radial distance from the tool axis (>= 0)
    double z = 0.0; // height along the tool axis from the tip
};

// Orthonormal frame for one tip of a segment:
//   z = tool axis — also the revolving axis for end caps
//   x = side axis from segment motion (orients the meridional plane)
//   y = z × x; meridian rests in the +y / +z half-plane
struct SegmentFrame
{
    vsg::dvec3 x{1.0, 0.0, 0.0};
    vsg::dvec3 y{0.0, 1.0, 0.0};
    vsg::dvec3 z{0.0, 0.0, 1.0};
};

vsg::dvec3 normalizeOr(const vsg::dvec3& v, const vsg::dvec3& fallback)
{
    const double len = vsg::length(v);
    if (len <= 1.0e-12) return fallback;
    return v / len;
}

SegmentFrame makeSegmentFrame(const ToolPose& pose, const vsg::dvec3& motion)
{
    SegmentFrame frame;
    frame.z = normalizeOr(pose.direction, vsg::dvec3(0.0, 0.0, 1.0));

    // Side axis from motion so the meridian faces consistently along the path.
    // Caps revolve around frame.z (tool axis), not this vector.
    frame.x = vsg::cross(motion, frame.z);
    if (vsg::length(frame.x) <= 1.0e-12)
    {
        vsg::dvec3 up(0.0, 0.0, 1.0);
        if (std::abs(vsg::dot(frame.z, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);
        frame.x = vsg::cross(up, frame.z);
    }
    frame.x = normalizeOr(frame.x, vsg::dvec3(1.0, 0.0, 0.0));
    frame.y = vsg::cross(frame.z, frame.x);
    return frame;
}

vsg::dvec3 transformLocal(const ToolPose& pose,
                          const SegmentFrame& frame,
                          const vsg::dvec3& local)
{
    return pose.position + frame.x * local.x + frame.y * local.y + frame.z * local.z;
}

// Meridional cutter profile in the (r, z) half-plane. Point count is fixed per
// tool type so each appendSegment costs the same regardless of path length.
std::vector<ProfilePoint> toolProfile(ToolType type, double radius, double height)
{
    std::vector<ProfilePoint> profile;
    if (radius <= 0.0) return profile;

    switch (type)
    {
    case ToolType::FlatNose:
        profile.push_back({0.0, 0.0});
        profile.push_back({radius, 0.0});
        profile.push_back({radius, height});
        profile.push_back({0.0, height});
        break;

    case ToolType::BallNose:
    {
        // Hemispherical tip of `radius`, then a cylindrical shank.
        constexpr int samples = 8;
        profile.reserve(samples + 3);
        for (int s = 0; s <= samples; ++s)
        {
            const double theta =
                pi * (1.0 - 0.5 * static_cast<double>(s) / static_cast<double>(samples));
            profile.push_back({radius * std::sin(theta), radius * (1.0 - std::cos(theta))});
        }
        profile.push_back({radius, radius + height});
        profile.push_back({0.0, radius + height});
        break;
    }

    case ToolType::Sphere:
    {
        constexpr int samples = 8;
        profile.reserve(samples + 1);
        for (int s = 0; s <= samples; ++s)
        {
            const double theta = pi * (1.0 - static_cast<double>(s) / static_cast<double>(samples));
            profile.push_back({radius * std::sin(theta), radius * (1.0 - std::cos(theta))});
        }
        break;
    }

    case ToolType::BullNose:
    {
        const double fillet = std::min(radius * 0.35, radius * 0.999);
        const double flatR = radius - fillet;
        profile.push_back({0.0, 0.0});
        if (flatR > 1.0e-9) profile.push_back({flatR, 0.0});

        constexpr int filletSamples = 8;
        for (int s = 1; s <= filletSamples; ++s)
        {
            const double phi = (0.5 * pi) * (static_cast<double>(s) / static_cast<double>(filletSamples));
            profile.push_back({flatR + fillet * std::sin(phi), fillet * (1.0 - std::cos(phi))});
        }
        profile.push_back({radius, fillet + height});
        profile.push_back({0.0, fillet + height});
        break;
    }

    case ToolType::None:
        break;
    }

    return profile;
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

// Profile sample spun around the tool axis (frame.z) — used for loft rings.
// Meridian rests in the +y / +z half-plane: local (0, r, z).
vsg::dvec3 profileAroundToolAxis(const ProfilePoint& p, double theta)
{
    return vsg::dvec3(p.r * std::sin(theta), p.r * std::cos(theta), p.z);
}

// Cap revolve: same axis as the tool axis (local z / frame.z). Starts at
// (0, r, z); rotation about +z by `angle` (sign set by flipWinding).
vsg::dvec3 profileAroundRevolvingAxis(const ProfilePoint& p, double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return vsg::dvec3(p.r * s, p.r * c, p.z);
}

void appendProfileLoft(TriangleMesh& mesh,
                       const ToolPose& tipA,
                       const ToolPose& tipB,
                       const SegmentFrame& frameA,
                       const SegmentFrame& frameB,
                       const std::vector<ProfilePoint>& profile,
                       int azimuthSegments)
{
    if (profile.empty() || azimuthSegments < 3) return;

    const auto nProfile = static_cast<int>(profile.size());

    auto at = [&](const ToolPose& tip, const SegmentFrame& frame, int i, int k) {
        const ProfilePoint& p = profile[static_cast<std::size_t>(i)];
        const double theta =
            (2.0 * pi * static_cast<double>(k)) / static_cast<double>(azimuthSegments);
        return transformLocal(tip, frame, profileAroundToolAxis(p, theta));
    };

    // Motion loft: each azimuthal edge of each profile sample travels A → B.
    for (int i = 0; i < nProfile; ++i)
    {
        if (profile[static_cast<std::size_t>(i)].r <= 1.0e-12) continue;
        for (int k = 0; k < azimuthSegments; ++k)
        {
            const int k1 = (k + 1) % azimuthSegments;
            stitchQuad(mesh,
                       at(tipA, frameA, i, k), at(tipA, frameA, i, k1),
                       at(tipB, frameB, i, k1), at(tipB, frameB, i, k));
        }
    }

    // Motion loft of the bands between consecutive profile samples.
    for (int i = 0; i + 1 < nProfile; ++i)
    {
        for (int k = 0; k < azimuthSegments; ++k)
        {
            stitchQuad(mesh,
                       at(tipA, frameA, i, k), at(tipA, frameA, i + 1, k),
                       at(tipB, frameB, i + 1, k), at(tipB, frameB, i, k));
        }
    }
}

// Close one tip by revolving the meridional profile 180° around the tool axis
// (frame.z). flipWinding selects both the revolve sense (+180° vs -180°) and
// the triangle winding.
void appendProfileCap(TriangleMesh& mesh,
                      const ToolPose& tip,
                      const SegmentFrame& frame,
                      const std::vector<ProfilePoint>& profile,
                      int steps,
                      bool flipWinding)
{
    if (profile.empty() || steps < 1) return;

    const auto nProfile = static_cast<int>(profile.size());
    std::vector<std::vector<vsg::dvec3>> rings(static_cast<std::size_t>(steps + 1));
    const double sense = flipWinding ? -1.0 : 1.0;

    for (int s = 0; s <= steps; ++s)
    {
        const double angle =
            sense * pi * (static_cast<double>(s) / static_cast<double>(steps));
        auto& ring = rings[static_cast<std::size_t>(s)];
        ring.reserve(static_cast<std::size_t>(nProfile));
        for (int i = 0; i < nProfile; ++i)
        {
            const ProfilePoint& p = profile[static_cast<std::size_t>(i)];
            ring.push_back(transformLocal(tip, frame, profileAroundRevolvingAxis(p, angle)));
        }
    }

    for (int s = 0; s < steps; ++s)
    {
        const auto& a = rings[static_cast<std::size_t>(s)];
        const auto& b = rings[static_cast<std::size_t>(s + 1)];
        for (int i = 0; i + 1 < nProfile; ++i)
        {
            if (flipWinding)
                stitchQuad(mesh, a[static_cast<std::size_t>(i)],
                           b[static_cast<std::size_t>(i)],
                           b[static_cast<std::size_t>(i + 1)],
                           a[static_cast<std::size_t>(i + 1)]);
            else
                stitchQuad(mesh, a[static_cast<std::size_t>(i)],
                           a[static_cast<std::size_t>(i + 1)],
                           b[static_cast<std::size_t>(i + 1)],
                           b[static_cast<std::size_t>(i)]);
        }
    }
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
    if (circleSegments < 3) circleSegments = 3;

    const std::vector<ProfilePoint> profile =
        toolProfile(type, static_cast<double>(radius), static_cast<double>(height));
    if (profile.empty()) return;

    const vsg::dvec3 motion = tipB.position - tipA.position;
    const SegmentFrame frameA = makeSegmentFrame(tipA, motion);
    const SegmentFrame frameB = makeSegmentFrame(tipB, motion);

    // 1) Rings around the tool axis at each tip, lofted A → B.
    appendProfileLoft(_mesh, tipA, tipB, frameA, frameB, profile, circleSegments);

    // 2) Caps: revolve the same profile 180° around the tool axis (frame.z).
    appendProfileCap(_mesh, tipA, frameA, profile, capRevolveSteps, false);
    appendProfileCap(_mesh, tipB, frameB, profile, capRevolveSteps, true);

    rebuildBvh();
    _lastPose = tipB;
}

} // namespace app
