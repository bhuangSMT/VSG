#include "ToolGeometry.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace app
{
namespace
{

constexpr float pi = 3.14159265358979323846f;

vsg::vec3 unit(float x, float y, float z)
{
    const float len = std::sqrt(x * x + y * y + z * z);
    if (len <= 0.0f) return vsg::vec3(0.0f, 0.0f, 1.0f);
    return vsg::vec3(x / len, y / len, z / len);
}

void addTriangle(TriangleMesh& mesh,
                 const vsg::vec3& a, const vsg::vec3& b, const vsg::vec3& c,
                 const vsg::vec3& normal)
{
    MeshTriangle tri;
    tri.normal = normal;
    tri.v0 = a;
    tri.v1 = b;
    tri.v2 = c;
    mesh.triangles.push_back(tri);
}

void addTriangleAuto(TriangleMesh& mesh, const vsg::vec3& a, const vsg::vec3& b, const vsg::vec3& c)
{
    const vsg::vec3 ab = b - a;
    const vsg::vec3 ac = c - a;
    addTriangle(mesh, a, b, c, unit(ab.y * ac.z - ab.z * ac.y,
                                    ab.z * ac.x - ab.x * ac.z,
                                    ab.x * ac.y - ab.y * ac.x));
}

// Ring of points on a circle in the XY plane at height z.
void ring(std::vector<vsg::vec3>& out, int segments, float radius, float z)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i)
    {
        const float a = (2.0f * pi * static_cast<float>(i)) / static_cast<float>(segments);
        out.push_back(vsg::vec3(radius * std::cos(a), radius * std::sin(a), z));
    }
}

void stitchRings(TriangleMesh& mesh,
                 const std::vector<vsg::vec3>& lower,
                 const std::vector<vsg::vec3>& upper,
                 bool outward)
{
    const int n = static_cast<int>(lower.size());
    for (int i = 0; i < n; ++i)
    {
        const int j = (i + 1) % n;
        if (outward)
        {
            addTriangleAuto(mesh, lower[i], lower[j], upper[j]);
            addTriangleAuto(mesh, lower[i], upper[j], upper[i]);
        }
        else
        {
            addTriangleAuto(mesh, lower[i], upper[j], lower[j]);
            addTriangleAuto(mesh, lower[i], upper[i], upper[j]);
        }
    }
}

void capDisk(TriangleMesh& mesh, const std::vector<vsg::vec3>& rim, bool upward)
{
    const vsg::vec3 centre(0.0f, 0.0f, rim.front().z);
    const vsg::vec3 normal = upward ? vsg::vec3(0.0f, 0.0f, 1.0f) : vsg::vec3(0.0f, 0.0f, -1.0f);
    const int n = static_cast<int>(rim.size());
    for (int i = 0; i < n; ++i)
    {
        const int j = (i + 1) % n;
        if (upward)
            addTriangle(mesh, centre, rim[i], rim[j], normal);
        else
            addTriangle(mesh, centre, rim[j], rim[i], normal);
    }
}

TriangleMesh sphereTool(float radius, int slices, int stacks)
{
    TriangleMesh mesh;
    mesh.name = "sphere";

    // Tip at the origin: the south pole sits on the surface, the centre is at
    // (0, 0, radius), and the rest of the ball rises along +Z.
    const vsg::vec3 centre(0.0f, 0.0f, radius);

    std::vector<std::vector<vsg::vec3>> rings(static_cast<std::size_t>(stacks + 1));
    for (int s = 0; s <= stacks; ++s)
    {
        // s = 0 is the tip (theta = pi), s = stacks is the north pole.
        const float theta = pi * (1.0f - static_cast<float>(s) / static_cast<float>(stacks));
        const float sinT = std::sin(theta);
        const float cosT = std::cos(theta);
        const float z = centre.z + radius * cosT;
        const float r = radius * sinT;

        auto& row = rings[static_cast<std::size_t>(s)];
        if (s == 0 || s == stacks)
        {
            row.push_back(vsg::vec3(0.0f, 0.0f, z));
            continue;
        }
        ring(row, slices, r, z);
    }

    for (int s = 0; s < stacks; ++s)
    {
        const auto& a = rings[static_cast<std::size_t>(s)];
        const auto& b = rings[static_cast<std::size_t>(s + 1)];

        if (a.size() == 1 && b.size() > 1)
        {
            for (int i = 0; i < slices; ++i)
            {
                const int j = (i + 1) % slices;
                addTriangleAuto(mesh, a[0], b[j], b[i]);
            }
        }
        else if (b.size() == 1 && a.size() > 1)
        {
            for (int i = 0; i < slices; ++i)
            {
                const int j = (i + 1) % slices;
                addTriangleAuto(mesh, a[i], a[j], b[0]);
            }
        }
        else
        {
            stitchRings(mesh, a, b, true);
        }
    }

    return mesh;
}

TriangleMesh flatNoseTool(float radius, float height, int slices)
{
    TriangleMesh mesh;
    mesh.name = "flat-nose";

    std::vector<vsg::vec3> bottom;
    std::vector<vsg::vec3> top;
    ring(bottom, slices, radius, 0.0f);
    ring(top, slices, radius, height);

    capDisk(mesh, bottom, false);
    stitchRings(mesh, bottom, top, true);
    capDisk(mesh, top, true);

    return mesh;
}

// Ring about an axis parallel to +X through (0, 0, axisZ).
void ringAboutX(std::vector<vsg::vec3>& out, int segments, float radius, float x, float axisZ)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(segments));
    for (int i = 0; i < segments; ++i)
    {
        const float a = (2.0f * pi * static_cast<float>(i)) / static_cast<float>(segments);
        // a = 0 → toward −Z (vertex side of the triangle).
        out.push_back(vsg::vec3(x, radius * std::sin(a), axisZ - radius * std::cos(a)));
    }
}

void stitchRevolveRings(TriangleMesh& mesh,
                        const std::vector<vsg::vec3>& a,
                        const std::vector<vsg::vec3>& b,
                        int slices)
{
    if (a.size() == 1 && b.size() > 1)
    {
        for (int i = 0; i < slices; ++i)
        {
            const int j = (i + 1) % slices;
            addTriangleAuto(mesh, a[0], b[j], b[i]);
        }
    }
    else if (b.size() == 1 && a.size() > 1)
    {
        for (int i = 0; i < slices; ++i)
        {
            const int j = (i + 1) % slices;
            addTriangleAuto(mesh, a[i], a[j], b[0]);
        }
    }
    else if (a.size() > 1 && b.size() > 1)
    {
        stitchRings(mesh, a, b, true);
    }
}

// Revolve an (x, r) polyline about +X through (0, 0, axisZ).
TriangleMesh revolveProfileX(const std::vector<std::pair<float, float>>& corners,
                             float axisZ, int slices, int stacks, const char* name)
{
    TriangleMesh mesh;
    mesh.name = name;
    if (corners.size() < 2 || slices < 3) return mesh;
    if (stacks < 1) stacks = 1;

    std::vector<std::pair<float, float>> samples;
    samples.reserve(corners.size() * static_cast<std::size_t>(stacks + 1));
    for (std::size_t e = 0; e + 1 < corners.size(); ++e)
    {
        const auto a = corners[e];
        const auto b = corners[e + 1];
        for (int s = 0; s < stacks; ++s)
        {
            const float t = static_cast<float>(s) / static_cast<float>(stacks);
            samples.push_back({a.first + (b.first - a.first) * t,
                               a.second + (b.second - a.second) * t});
        }
    }
    samples.push_back(corners.back());

    std::vector<std::vector<vsg::vec3>> rings(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const float x = samples[i].first;
        const float r = samples[i].second;
        auto& row = rings[i];
        if (r <= 1.0e-8f)
            row.push_back(vsg::vec3(x, 0.0f, axisZ));
        else
            ringAboutX(row, slices, r, x, axisZ);
    }

    for (std::size_t s = 0; s + 1 < rings.size(); ++s)
        stitchRevolveRings(mesh, rings[s], rings[s + 1], slices);

    return mesh;
}

// Trapezoid + rectangle cutter. Origin = shoulder-rectangle midpoint (CL).
// Spindle +X on the shank edge; outer rim toward −Z. a=0 at the shoulder mid
// sits at the mesh origin (axisZ = shankRadius + H2/2).
TriangleMesh grindingWheelTool(const GrindingWheelProfile& wheel, float shankRadius,
                               int slices, int stacks)
{
    TriangleMesh mesh;
    mesh.name = "grinding-wheel";
    if (!wheel.valid() || slices < 3) return mesh;
    if (!(shankRadius >= 0.0f)) shankRadius = 0.0f;

    const float halfA = 0.5f * wheel.tipWidth;
    const float halfB = 0.5f * wheel.shoulderWidth;
    const float h2 = (wheel.shoulderHeight > 0.0f) ? wheel.shoulderHeight : 0.0f;
    const float axisZ = shankRadius + 0.5f * h2;
    const float rInner = shankRadius;
    const float rJunc = shankRadius + h2;
    const float rTip = shankRadius + wheel.totalHeight();
    const bool hasRect = h2 > 1.0e-8f;
    const bool flared = std::abs(wheel.tipWidth - wheel.shoulderWidth) > 1.0e-8f;

    std::vector<std::pair<float, float>> corners;
    corners.push_back({-halfB, rInner});
    if (hasRect && flared) corners.push_back({-halfB, rJunc});
    corners.push_back({-halfA, rTip});
    corners.push_back({halfA, rTip});
    if (hasRect && flared) corners.push_back({halfB, rJunc});
    corners.push_back({halfB, rInner});
    return revolveProfileX(corners, axisZ, slices, stacks, "grinding-wheel");
}

// Ball-end mill: hemispherical tip of `radius` (tip at the origin, centre at
// z = radius), then a cylindrical shank of the same radius and length `height`.
TriangleMesh ballNoseTool(float radius, float height, int slices, int stacks)
{
    TriangleMesh mesh;
    mesh.name = "ball-nose";

    if (stacks < 2) stacks = 2;
    const vsg::vec3 centre(0.0f, 0.0f, radius);

    // Hemisphere only: s = 0 at the tip, s = stacks at the equator.
    std::vector<std::vector<vsg::vec3>> rings(static_cast<std::size_t>(stacks + 1));
    for (int s = 0; s <= stacks; ++s)
    {
        const float theta = pi * (1.0f - 0.5f * static_cast<float>(s) / static_cast<float>(stacks));
        const float sinT = std::sin(theta);
        const float cosT = std::cos(theta);
        const float z = centre.z + radius * cosT;
        const float r = radius * sinT;

        auto& row = rings[static_cast<std::size_t>(s)];
        if (s == 0)
        {
            row.push_back(vsg::vec3(0.0f, 0.0f, z));
            continue;
        }
        ring(row, slices, r, z);
    }

    for (int s = 0; s < stacks; ++s)
    {
        const auto& a = rings[static_cast<std::size_t>(s)];
        const auto& b = rings[static_cast<std::size_t>(s + 1)];

        if (a.size() == 1 && b.size() > 1)
        {
            for (int i = 0; i < slices; ++i)
            {
                const int j = (i + 1) % slices;
                addTriangleAuto(mesh, a[0], b[j], b[i]);
            }
        }
        else
        {
            stitchRings(mesh, a, b, true);
        }
    }

    std::vector<vsg::vec3> top;
    ring(top, slices, radius, radius + height);
    stitchRings(mesh, rings[static_cast<std::size_t>(stacks)], top, true);
    capDisk(mesh, top, true);

    return mesh;
}

TriangleMesh bullNoseTool(float radius, float height, float cornerRadius, int slices, int filletStacks)
{
    TriangleMesh mesh;
    mesh.name = "bull-nose";

    const float r = std::min(cornerRadius, radius * 0.999f);
    const float flatRadius = radius - r;

    // Flat tip disk at z = 0, then a quarter-torus fillet up to z = r, then a
    // straight cylinder of length `height` above that.
    std::vector<vsg::vec3> flatRim;
    ring(flatRim, slices, flatRadius, 0.0f);
    if (flatRadius > 1.0e-5f) capDisk(mesh, flatRim, false);

    std::vector<vsg::vec3> prev = flatRim;
    if (flatRadius <= 1.0e-5f)
    {
        // Degenerates to a ball-nose when the flat vanishes.
        prev.clear();
        prev.push_back(vsg::vec3(0.0f, 0.0f, 0.0f));
    }

    for (int s = 1; s <= filletStacks; ++s)
    {
        const float phi = (0.5f * pi) * (static_cast<float>(s) / static_cast<float>(filletStacks));
        const float z = r - r * std::cos(phi);
        const float ringR = flatRadius + r * std::sin(phi);

        std::vector<vsg::vec3> next;
        if (ringR <= 1.0e-5f)
            next.push_back(vsg::vec3(0.0f, 0.0f, z));
        else
            ring(next, slices, ringR, z);

        if (prev.size() == 1 && next.size() > 1)
        {
            for (int i = 0; i < slices; ++i)
            {
                const int j = (i + 1) % slices;
                addTriangleAuto(mesh, prev[0], next[j], next[i]);
            }
        }
        else
        {
            stitchRings(mesh, prev, next, true);
        }
        prev = std::move(next);
    }

    std::vector<vsg::vec3> top;
    ring(top, slices, radius, r + height);
    stitchRings(mesh, prev, top, true);
    capDisk(mesh, top, true);

    return mesh;
}

} // namespace

TriangleMesh createShankMesh(float radius, float z0, float height, int slices)
{
    TriangleMesh mesh;
    mesh.name = "shank";
    if (!(radius > 0.0f) || !(height > 0.0f)) return mesh;
    if (slices < 3) slices = 3;

    std::vector<vsg::vec3> bottom;
    std::vector<vsg::vec3> top;
    ring(bottom, slices, radius, z0);
    ring(top, slices, radius, z0 + height);
    capDisk(mesh, bottom, false);
    stitchRings(mesh, bottom, top, true);
    capDisk(mesh, top, true);
    return mesh;
}

TriangleMesh createGrindingShankMesh(float shoulderHeight, float shankRadius, float shankLength,
                                     int slices, int stacks)
{
    if (!(shankRadius > 0.0f) || !(shankLength > 0.0f)) return {};
    if (slices < 3) slices = 3;
    if (stacks < 1) stacks = 1;

    const float h2 = (shoulderHeight > 0.0f) ? shoulderHeight : 0.0f;
    const float halfLen = 0.5f * shankLength;
    const float axisZ = shankRadius + 0.5f * h2;
    const std::vector<std::pair<float, float>> corners = {
        {-halfLen, 0.0f},
        {-halfLen, shankRadius},
        {halfLen, shankRadius},
        {halfLen, 0.0f},
    };
    return revolveProfileX(corners, axisZ, slices, stacks, "shank");
}

TriangleMesh createToolMesh(ToolType type, float radius, float height,
                            int slices, int stacks, int filletStacks,
                            float vertexAngleDeg, float shankRadius, float shankLength,
                            GrindingWheelProfile wheel)
{
    if (slices < 3) slices = 3;
    if (stacks < 2) stacks = 2;
    if (filletStacks < 1) filletStacks = 1;
    (void)shankLength;
    (void)vertexAngleDeg;

    switch (type)
    {
    case ToolType::Sphere:
        return sphereTool(radius, slices, stacks);
    case ToolType::FlatNose:
        return flatNoseTool(radius, height, slices);
    case ToolType::BallNose:
        return ballNoseTool(radius, height, slices, stacks);
    case ToolType::BullNose:
        return bullNoseTool(radius, height, bullNoseFilletRadius(radius), slices, filletStacks);
    case ToolType::GrindingWheel:
        if (!wheel.valid()) wheel = GrindingWheelProfile{};
        return grindingWheelTool(wheel, shankRadius, slices, stacks);
    case ToolType::None:
        break;
    }

    return TriangleMesh{};
}

} // namespace app
