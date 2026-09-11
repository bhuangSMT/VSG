#include "ModelPick.h"

#include <cmath>
#include <cstdint>

namespace app
{
namespace
{

bool intersectTriangle(const vsg::dvec3& origin, const vsg::dvec3& direction,
                       const vsg::dvec3& v0, const vsg::dvec3& v1, const vsg::dvec3& v2,
                       double tMin, double tMax,
                       double& t, vsg::dvec3& normal)
{
    const vsg::dvec3 e1 = v1 - v0;
    const vsg::dvec3 e2 = v2 - v0;
    const vsg::dvec3 p = vsg::cross(direction, e2);
    const double det = vsg::dot(e1, p);

    if (std::abs(det) < 1.0e-12) return false;

    const double invDet = 1.0 / det;
    const vsg::dvec3 s = origin - v0;
    const double u = vsg::dot(s, p) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    const vsg::dvec3 q = vsg::cross(s, e1);
    const double v = vsg::dot(direction, q) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;

    const double hitT = vsg::dot(e2, q) * invDet;
    if (hitT < tMin || hitT > tMax) return false;

    t = hitT;
    normal = vsg::cross(e1, e2);
    const double len = vsg::length(normal);
    if (len > 0.0) normal /= len;
    else normal = vsg::dvec3(0.0, 0.0, 1.0);

    // Face the side the ray came from so the tool axis points back at the camera.
    if (vsg::dot(normal, direction) > 0.0) normal = -normal;

    return true;
}

vsg::dvec3 toD(const vsg::vec3& v)
{
    return vsg::dvec3(v.x, v.y, v.z);
}

} // namespace

std::optional<ModelHit> pickBRep(const BRep& brep,
                                 const vsg::dvec3& origin,
                                 const vsg::dvec3& direction,
                                 double tMin,
                                 double tMax)
{
    if (brep.bvh().empty() || brep.faceCount() == 0) return std::nullopt;

    double invDir[3]{};
    double originA[3]{origin.x, origin.y, origin.z};
    for (int i = 0; i < 3; ++i)
    {
        invDir[i] = (std::abs(direction[i]) > 1.0e-15)
                        ? 1.0 / direction[i]
                        : std::copysign(1.0e15, direction[i]);
    }

    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();

    ModelHit best;
    bool found = false;
    double bestT = tMax;

    brep.bvh().queryRay(originA, invDir, tMin, tMax, [&](std::size_t f) {
        const std::uint32_t begin = faceOffsets[f];
        if (faceOffsets[f + 1] - begin < 3) return;

        const vsg::dvec3 a = toD(verts[faceVertices[begin]]);
        const vsg::dvec3 b = toD(verts[faceVertices[begin + 1]]);
        const vsg::dvec3 c = toD(verts[faceVertices[begin + 2]]);

        double t = 0.0;
        vsg::dvec3 normal;
        if (!intersectTriangle(origin, direction, a, b, c, tMin, bestT, t, normal)) return;

        bestT = t;
        best.t = t;
        best.position = origin + direction * t;
        best.normal = normal;
        found = true;
    });

    if (!found) return std::nullopt;
    return best;
}

std::optional<ModelHit> pickPlane(const vsg::dvec3& origin,
                                  const vsg::dvec3& direction,
                                  const vsg::dvec3& planePoint,
                                  const vsg::dvec3& planeNormal,
                                  double tMin,
                                  double tMax)
{
    const double denom = vsg::dot(direction, planeNormal);
    if (std::abs(denom) < 1.0e-15) return std::nullopt;

    const double t = vsg::dot(planePoint - origin, planeNormal) / denom;
    if (t < tMin || t > tMax) return std::nullopt;

    ModelHit hit;
    hit.t = t;
    hit.position = origin + direction * t;

    const double nLen = vsg::length(planeNormal);
    hit.normal = (nLen > 0.0) ? planeNormal / nLen : vsg::dvec3(0.0, 0.0, 1.0);
    if (vsg::dot(hit.normal, direction) > 0.0) hit.normal = -hit.normal;

    return hit;
}

} // namespace app
