#include "WorldAxes.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace app
{
namespace
{

constexpr float pi = 3.14159265358979323846f;

vsg::dvec3 axisDir(int axisIndex)
{
    if (axisIndex == 0) return {1.0, 0.0, 0.0};
    if (axisIndex == 1) return {0.0, 1.0, 0.0};
    return {0.0, 0.0, 1.0};
}

vsg::vec3 unitPerp(const vsg::vec3& axis)
{
    const vsg::vec3 hint = (std::abs(axis.z) < 0.9f) ? vsg::vec3(0.0f, 0.0f, 1.0f)
                                                     : vsg::vec3(1.0f, 0.0f, 0.0f);
    return vsg::normalize(vsg::cross(axis, hint));
}

void addTri(TriangleMesh& mesh, const vsg::vec3& a, const vsg::vec3& b, const vsg::vec3& c)
{
    MeshTriangle t;
    t.v0 = a;
    t.v1 = b;
    t.v2 = c;
    const vsg::vec3 n = vsg::cross(b - a, c - a);
    const float len = vsg::length(n);
    t.normal = (len > 0.0f) ? n / len : vsg::vec3(0.0f, 0.0f, 1.0f);
    mesh.triangles.push_back(t);
}

void addTubeMesh(TriangleMesh& mesh, const vsg::vec3& a, const vsg::vec3& b, float radius,
                 int slices)
{
    vsg::vec3 axis = b - a;
    const float len = vsg::length(axis);
    if (len <= 0.0f || !(radius > 0.0f) || slices < 3) return;
    axis /= len;
    const vsg::vec3 side = unitPerp(axis);
    const vsg::vec3 up = vsg::normalize(vsg::cross(axis, side));

    std::vector<vsg::vec3> ringA(static_cast<std::size_t>(slices));
    std::vector<vsg::vec3> ringB(static_cast<std::size_t>(slices));
    for (int i = 0; i < slices; ++i)
    {
        const float ang = (2.0f * pi * static_cast<float>(i)) / static_cast<float>(slices);
        const vsg::vec3 n = side * std::cos(ang) + up * std::sin(ang);
        ringA[static_cast<std::size_t>(i)] = a + n * radius;
        ringB[static_cast<std::size_t>(i)] = b + n * radius;
    }
    for (int i = 0; i < slices; ++i)
    {
        const int j = (i + 1) % slices;
        addTri(mesh, ringA[i], ringA[j], ringB[j]);
        addTri(mesh, ringA[i], ringB[j], ringB[i]);
    }
}

void addConeMesh(TriangleMesh& mesh, const vsg::vec3& base, const vsg::vec3& tip, float radius,
                 int slices)
{
    vsg::vec3 axis = tip - base;
    const float len = vsg::length(axis);
    if (len <= 0.0f || !(radius > 0.0f) || slices < 3) return;
    axis /= len;
    const vsg::vec3 side = unitPerp(axis);
    const vsg::vec3 up = vsg::normalize(vsg::cross(axis, side));

    std::vector<vsg::vec3> ring(static_cast<std::size_t>(slices));
    for (int i = 0; i < slices; ++i)
    {
        const float ang = (2.0f * pi * static_cast<float>(i)) / static_cast<float>(slices);
        const vsg::vec3 n = side * std::cos(ang) + up * std::sin(ang);
        ring[static_cast<std::size_t>(i)] = base + n * radius;
    }
    for (int i = 0; i < slices; ++i)
    {
        const int j = (i + 1) % slices;
        addTri(mesh, tip, ring[i], ring[j]);
        addTri(mesh, base, ring[j], ring[i]); // base cap
    }
}

bool intersectTube(const vsg::dvec3& origin, const vsg::dvec3& dir, const vsg::dvec3& a,
                   const vsg::dvec3& b, double radius, double& tHit)
{
    vsg::dvec3 axis = b - a;
    const double axisLen = vsg::length(axis);
    if (axisLen <= 0.0 || !(radius > 0.0)) return false;
    axis /= axisLen;

    const vsg::dvec3 m = origin - a;
    const vsg::dvec3 dPerp = dir - axis * vsg::dot(dir, axis);
    const vsg::dvec3 mPerp = m - axis * vsg::dot(m, axis);
    const double A = vsg::dot(dPerp, dPerp);
    if (A < 1.0e-12) return false;
    const double B = vsg::dot(dPerp, mPerp);
    const double C = vsg::dot(mPerp, mPerp) - radius * radius;
    const double disc = B * B - A * C;
    if (disc < 0.0) return false;
    const double t = (-B - std::sqrt(disc)) / A;
    if (t < 1.0e-6) return false;
    const vsg::dvec3 hit = origin + dir * t;
    const double along = vsg::dot(hit - a, axis);
    if (along < 0.0 || along > axisLen) return false;
    tHit = t;
    return true;
}

bool intersectCone(const vsg::dvec3& origin, const vsg::dvec3& dir, const vsg::dvec3& base,
                   const vsg::dvec3& tip, double radius, double& tHit)
{
    vsg::dvec3 axis = tip - base;
    const double height = vsg::length(axis);
    if (height <= 0.0 || !(radius > 0.0)) return false;
    axis /= height;
    const double k = radius / height;

    const vsg::dvec3 v = origin - tip;
    const double dv = vsg::dot(dir, axis);
    const double vv = vsg::dot(v, axis);
    const vsg::dvec3 dPerp = dir - axis * dv;
    const vsg::dvec3 vPerp = v - axis * vv;

    const double A = vsg::dot(dPerp, dPerp) - k * k * dv * dv;
    const double B = vsg::dot(dPerp, vPerp) - k * k * dv * vv;
    const double C = vsg::dot(vPerp, vPerp) - k * k * vv * vv;
    const double disc = B * B - A * C;
    if (disc < 0.0 || std::abs(A) < 1.0e-12) return false;

    auto tryT = [&](double t) {
        if (t < 1.0e-6) return false;
        const vsg::dvec3 hit = origin + dir * t;
        const double along = vsg::dot(tip - hit, axis); // 0 at tip, height at base
        if (along < 0.0 || along > height) return false;
        tHit = t;
        return true;
    };

    const double sqrtDisc = std::sqrt(disc);
    const double t0 = (-B - sqrtDisc) / A;
    const double t1 = (-B + sqrtDisc) / A;
    if (tryT(t0)) return true;
    if (tryT(t1)) return true;
    return false;
}

} // namespace

vsg::vec4 worldAxisColor(int axisIndex, bool selected)
{
    if (selected)
    {
        if (axisIndex == 0) return {1.0f, 0.55f, 0.15f, 1.0f};
        if (axisIndex == 1) return {0.35f, 1.0f, 0.35f, 1.0f};
        return {0.35f, 0.65f, 1.0f, 1.0f};
    }
    if (axisIndex == 0) return {0.90f, 0.15f, 0.12f, 1.0f};
    if (axisIndex == 1) return {0.15f, 0.75f, 0.20f, 1.0f};
    return {0.20f, 0.40f, 0.95f, 1.0f};
}

std::array<TriangleMesh, 6> buildWorldAxesMeshes(const WorldAxesSpec& spec,
                                                 WorldAxisPart selected)
{
    std::array<TriangleMesh, 6> meshes{};
    const float tubeLen = std::max(spec.length - spec.coneLength, spec.length * 0.55f);
    constexpr int slices = 16;

    for (int axis = 0; axis < 3; ++axis)
    {
        const vsg::vec3 dir(static_cast<float>(axisDir(axis).x),
                            static_cast<float>(axisDir(axis).y),
                            static_cast<float>(axisDir(axis).z));
        const vsg::vec3 origin(0.0f, 0.0f, 0.0f);
        const vsg::vec3 tubeEnd = dir * tubeLen;
        const vsg::vec3 tip = dir * spec.length;

        // Selecting either tube or cone highlights the whole axis.
        const bool axisSel = worldAxisIndex(selected) == axis;

        meshes[static_cast<std::size_t>(axis * 2)].name = "axis-tube";
        meshes[static_cast<std::size_t>(axis * 2 + 1)].name = "axis-cone";
        const float tubeR = axisSel ? spec.tubeRadius * 1.35f : spec.tubeRadius;
        const float coneR = axisSel ? spec.coneRadius * 1.25f : spec.coneRadius;
        addTubeMesh(meshes[static_cast<std::size_t>(axis * 2)], origin, tubeEnd, tubeR, slices);
        addConeMesh(meshes[static_cast<std::size_t>(axis * 2 + 1)], tubeEnd, tip, coneR, slices);
    }
    return meshes;
}

std::optional<WorldAxisPart> pickWorldAxes(const WorldAxesSpec& spec,
                                           const vsg::dvec3& origin,
                                           const vsg::dvec3& direction)
{
    const double dirLen = vsg::length(direction);
    if (dirLen <= 0.0) return std::nullopt;
    const vsg::dvec3 dir = direction / dirLen;

    const double tubeLen = std::max(static_cast<double>(spec.length - spec.coneLength),
                                    static_cast<double>(spec.length) * 0.55);

    double bestT = 1.0e9;
    WorldAxisPart best = WorldAxisPart::None;

    for (int axis = 0; axis < 3; ++axis)
    {
        const vsg::dvec3 aDir = axisDir(axis);
        const vsg::dvec3 tubeA(0.0, 0.0, 0.0);
        const vsg::dvec3 tubeB = aDir * tubeLen;
        const vsg::dvec3 tip = aDir * static_cast<double>(spec.length);

        double t = 0.0;
        if (intersectTube(origin, dir, tubeA, tubeB, spec.tubeRadius, t) && t < bestT)
        {
            bestT = t;
            best = static_cast<WorldAxisPart>(axis * 2);
        }
        if (intersectCone(origin, dir, tubeB, tip, spec.coneRadius, t) && t < bestT)
        {
            bestT = t;
            best = static_cast<WorldAxisPart>(axis * 2 + 1);
        }
    }

    if (best == WorldAxisPart::None) return std::nullopt;
    return best;
}

} // namespace app
