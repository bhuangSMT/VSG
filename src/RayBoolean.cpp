#include "RayBoolean.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <vsg/maths/transform.h>

#include "BoundingBox.h"
#include "BVH.h"
#include "TriangleMesh.h"

namespace app
{
namespace
{

struct Interval
{
    double lo = 0.0;
    double hi = 0.0;
    Normal3f loNormal{0.0f, 0.0f, 0.0f};
    Normal3f hiNormal{0.0f, 0.0f, 0.0f};
};

struct Hit
{
    double along = 0.0;
    Normal3f normal{0.0f, 0.0f, 0.0f};
};

// Query the SweptVolume's own mesh/BVH in world space. Fitting is uniform scale
// + translate, so model-space axis-aligned rays stay axis-aligned in world space
// and the existing BVH can be used without copying or rebuilding.
struct WorldSweep
{
    const TriangleMesh& mesh;
    const BVH& bvh;
    BoundingBox worldBounds;
    vsg::dmat4 modelToWorld{};
    vsg::dmat4 worldToModel{};
};

vsg::dvec3 transformNormal(const vsg::dmat4& m, const float n[3])
{
    const vsg::dvec4 d = m * vsg::dvec4(n[0], n[1], n[2], 0.0);
    vsg::dvec3 out(d.x, d.y, d.z);
    const double len = vsg::length(out);
    if (len > 0.0) out /= len;
    return out;
}

void sortAndMerge(std::vector<Hit>& hits, double tolerance)
{
    std::sort(hits.begin(), hits.end(),
              [](const Hit& lhs, const Hit& rhs) { return lhs.along < rhs.along; });

    auto last = std::unique(hits.begin(), hits.end(),
                            [tolerance](const Hit& lhs, const Hit& rhs) {
                                return std::abs(rhs.along - lhs.along) <= tolerance;
                            });
    hits.erase(last, hits.end());
}

Normal3f orientOutward(Normal3f normal, std::size_t axis, bool entering)
{
    const float along = normal[axis];
    const bool flip = entering ? (along > 0.0f) : (along < 0.0f);
    if (flip)
    {
        for (int i = 0; i < 3; ++i) normal[i] = -normal[i];
    }
    return normal;
}

Normal3f axisOutward(std::size_t axis, bool entering)
{
    Normal3f n{0.0f, 0.0f, 0.0f};
    n[axis] = entering ? -1.0f : 1.0f;
    return n;
}

Normal3f safeOrientOutward(Normal3f normal, std::size_t axis, bool entering)
{
    const float len2 = normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2];
    if (len2 < 1.0e-12f) return axisOutward(axis, entering);
    return orientOutward(normal, axis, entering);
}

// Axis-aligned hits against the world-space sweep BVH. (u0, v0) are model-space
// grid coordinates; hits come back in model space along `axis`.
void collectHits(const WorldSweep& sweep,
                 std::size_t axis, std::size_t u, std::size_t v,
                 double u0, double v0,
                 std::vector<Hit>& hits)
{
    hits.clear();
    if (sweep.bvh.empty()) return;

    vsg::dvec3 modelSample(0.0, 0.0, 0.0);
    modelSample[u] = u0;
    modelSample[v] = v0;
    const vsg::dvec3 worldSample = sweep.modelToWorld * modelSample;
    const double u0w = worldSample[u];
    const double v0w = worldSample[v];

    const auto& tris = sweep.mesh.triangles;

    sweep.bvh.query(u, v, u0w, v0w, [&](std::size_t f) {
        if (f >= tris.size()) return;
        const MeshTriangle& tri = tris[f];

        const double a[3]{tri.v0.x, tri.v0.y, tri.v0.z};
        const double b[3]{tri.v1.x, tri.v1.y, tri.v1.z};
        const double c[3]{tri.v2.x, tri.v2.y, tri.v2.z};

        const double au = a[u], av = a[v];
        const double bu = b[u], bv = b[v];
        const double cu = c[u], cv = c[v];

        const double loU = std::min({au, bu, cu});
        const double hiU = std::max({au, bu, cu});
        const double loV = std::min({av, bv, cv});
        const double hiV = std::max({av, bv, cv});
        if (u0w < loU || u0w > hiU || v0w < loV || v0w > hiV) return;

        const double denom = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
        const double areaScale = (hiU - loU) * (hiV - loV);
        if (std::abs(denom) <= 1e-12 * std::max(areaScale, 1e-300)) return;

        const double w0 = ((bv - cv) * (u0w - cu) + (cu - bu) * (v0w - cv)) / denom;
        const double w1 = ((cv - av) * (u0w - cu) + (au - cu) * (v0w - cv)) / denom;
        const double w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) return;

        const double alongWorld = w0 * a[axis] + w1 * b[axis] + w2 * c[axis];

        vsg::dvec3 worldHit;
        worldHit[u] = u0w;
        worldHit[v] = v0w;
        worldHit[axis] = alongWorld;
        const vsg::dvec3 modelHit = sweep.worldToModel * worldHit;

        Hit hit;
        hit.along = modelHit[axis];

        const double e1[3]{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3]{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double n[3]{e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};
        const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (length > 0.0)
        {
            const float nw[3]{static_cast<float>(n[0] / length),
                              static_cast<float>(n[1] / length),
                              static_cast<float>(n[2] / length)};
            const vsg::dvec3 nm = transformNormal(sweep.worldToModel, nw);
            hit.normal = {static_cast<float>(nm.x), static_cast<float>(nm.y),
                          static_cast<float>(nm.z)};
        }

        hits.push_back(hit);
    });
}

std::vector<Interval> intervalsFromHits(std::vector<Hit>& hits,
                                        std::size_t axis,
                                        double tolerance)
{
    sortAndMerge(hits, tolerance);

    std::vector<Interval> intervals;
    for (std::size_t i = 0; i + 1 < hits.size(); i += 2)
    {
        Interval iv;
        iv.lo = hits[i].along;
        iv.hi = hits[i + 1].along;
        if (!(iv.hi > iv.lo)) continue;
        iv.loNormal = safeOrientOutward(hits[i].normal, axis, true);
        iv.hiNormal = safeOrientOutward(hits[i + 1].normal, axis, false);
        intervals.push_back(iv);
    }
    return intervals;
}

bool chainMayHitSweep(const WorldSweep& sweep,
                      std::size_t axis,
                      const RayChain& chain)
{
    if (!sweep.worldBounds.valid() || chain.rays.empty()) return false;

    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const Point3d& sample = chain.rays.front().startPoint;
    const double u0 = sample[u];
    const double v0 = sample[v];

    double tMin = chain.rays.front().startPoint[axis];
    double tMax = chain.rays.front().endPoint[axis];
    for (const Ray& ray : chain.rays)
    {
        tMin = std::min(tMin, std::min(ray.startPoint[axis], ray.endPoint[axis]));
        tMax = std::max(tMax, std::max(ray.startPoint[axis], ray.endPoint[axis]));
    }

    vsg::dvec3 modelLo;
    modelLo[u] = u0;
    modelLo[v] = v0;
    modelLo[axis] = tMin;
    vsg::dvec3 modelHi = modelLo;
    modelHi[axis] = tMax;

    const vsg::dvec3 worldLo = sweep.modelToWorld * modelLo;
    const vsg::dvec3 worldHi = sweep.modelToWorld * modelHi;

    const double wu0 = worldLo[u];
    const double wv0 = worldLo[v];
    const double wtMin = std::min(worldLo[axis], worldHi[axis]);
    const double wtMax = std::max(worldLo[axis], worldHi[axis]);

    const Point3d& bmin = sweep.worldBounds.min();
    const Point3d& bmax = sweep.worldBounds.max();
    if (wu0 < bmin[u] || wu0 > bmax[u] || wv0 < bmin[v] || wv0 > bmax[v]) return false;
    return !(wtMax < bmin[axis] || wtMin > bmax[axis]);
}

std::vector<Interval> subtractIntervals(const Interval& solid,
                                        const std::vector<Interval>& cutters)
{
    std::vector<Interval> remaining{solid};
    for (const Interval& cut : cutters)
    {
        std::vector<Interval> next;
        for (const Interval& piece : remaining)
        {
            if (cut.hi <= piece.lo || cut.lo >= piece.hi)
            {
                next.push_back(piece);
                continue;
            }

            if (cut.lo > piece.lo + 1.0e-12)
            {
                Interval left = piece;
                left.hi = cut.lo;
                left.hiNormal = piece.hiNormal;
                next.push_back(left);
            }
            if (cut.hi < piece.hi - 1.0e-12)
            {
                Interval right = piece;
                right.lo = cut.hi;
                right.loNormal = piece.loNormal;
                next.push_back(right);
            }
        }
        remaining.swap(next);
    }
    return remaining;
}

std::vector<Interval> unionIntervals(const Interval& solid,
                                     const std::vector<Interval>& added)
{
    std::vector<Interval> all{solid};
    for (const Interval& cut : added)
    {
        Interval piece = cut;
        piece.loNormal = solid.loNormal;
        piece.hiNormal = solid.hiNormal;
        all.push_back(piece);
    }
    if (all.empty()) return all;

    std::sort(all.begin(), all.end(),
              [](const Interval& a, const Interval& b) { return a.lo < b.lo; });

    std::vector<Interval> merged;
    merged.push_back(all.front());
    for (std::size_t i = 1; i < all.size(); ++i)
    {
        Interval& last = merged.back();
        const Interval& cur = all[i];
        if (cur.lo <= last.hi + 1.0e-12)
        {
            if (cur.hi > last.hi)
            {
                last.hi = cur.hi;
                last.hiNormal = cur.hiNormal;
            }
        }
        else
        {
            merged.push_back(cur);
        }
    }
    return merged;
}

Ray makeRay(std::size_t axis, std::size_t u, std::size_t v,
            double along0, double along1,
            double u0, double v0,
            const Normal3f& n0, const Normal3f& n1,
            bool fromBoolean)
{
    Ray ray;
    ray.startPoint[axis] = along0;
    ray.startPoint[u] = u0;
    ray.startPoint[v] = v0;
    ray.endPoint[axis] = along1;
    ray.endPoint[u] = u0;
    ray.endPoint[v] = v0;
    ray.startNormal = n0;
    ray.endNormal = n1;
    ray.fromBoolean = fromBoolean;
    return ray;
}

RayChain processChain(std::size_t axis,
                      const RayChain& chain,
                      const WorldSweep& sweep,
                      BooleanOp op,
                      double mergeTol)
{
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;

    RayChain out;
    out.u = chain.u;
    out.v = chain.v;
    if (chain.rays.empty()) return out;

    const double u0 = chain.rays.front().startPoint[u];
    const double v0 = chain.rays.front().startPoint[v];

    std::vector<Hit> hits;
    collectHits(sweep, axis, u, v, u0, v0, hits);
    if (hits.empty())
    {
        out.rays = chain.rays;
        return out;
    }

    const std::vector<Interval> sweepSolid = intervalsFromHits(hits, axis, mergeTol);
    if (sweepSolid.empty())
    {
        out.rays = chain.rays;
        return out;
    }

    for (const Ray& ray : chain.rays)
    {
        const double t0 = ray.startPoint[axis];
        const double t1 = ray.endPoint[axis];
        if (!(t1 > t0)) continue;

        Interval solid;
        solid.lo = t0;
        solid.hi = t1;
        solid.loNormal = ray.startNormal;
        solid.hiNormal = ray.endNormal;

        std::vector<Interval> result;
        if (op == BooleanOp::Subtraction)
            result = subtractIntervals(solid, sweepSolid);
        else
            result = unionIntervals(solid, sweepSolid);

        for (const Interval& iv : result)
        {
            if (!(iv.hi > iv.lo)) continue;
            out.rays.push_back(makeRay(axis, u, v, iv.lo, iv.hi, u0, v0,
                                       ray.startNormal, ray.endNormal,
                                       true));
        }
    }

    return out;
}

} // namespace

RayModel applyBoolean(const RayModel& source,
                      const SweptVolume& sweep,
                      BooleanOp op,
                      const vsg::dmat4& modelToWorld)
{
    auto sourceLock = source.lockChains();

    RayModel result;
    result._bounds = source.bounds();
    result._resolution = source.resolution();

    if (op == BooleanOp::None || sweep.empty() || sweep.bvh().empty())
    {
        result._chains = source._chains;
        return result;
    }

    const WorldSweep worldSweep{sweep.mesh(), sweep.bvh(), sweep.bvh().bounds(),
                                modelToWorld, vsg::inverse(modelToWorld)};
    if (!worldSweep.worldBounds.valid())
    {
        result._chains = source._chains;
        return result;
    }

    const double mergeTol = std::max(1.0e-9, worldSweep.worldBounds.diagonal() * 1.0e-9);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const auto& inChains = source._chains[axis];
        std::vector<RayChain> outChains(inChains.size());
        std::vector<char> keep(inChains.size(), 0);

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, inChains.size()),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i != range.end(); ++i)
                {
                    const RayChain& chain = inChains[i];
                    if (!chainMayHitSweep(worldSweep, axis, chain))
                    {
                        outChains[i] = chain;
                        keep[i] = 1;
                        continue;
                    }

                    RayChain updated = processChain(axis, chain, worldSweep, op, mergeTol);
                    if (!updated.rays.empty())
                    {
                        outChains[i] = std::move(updated);
                        keep[i] = 1;
                    }
                }
            });

        std::vector<RayChain> compacted;
        compacted.reserve(inChains.size());
        for (std::size_t i = 0; i < inChains.size(); ++i)
        {
            if (keep[i]) compacted.push_back(std::move(outChains[i]));
        }
        result._chains[axis] = std::move(compacted);
    }

    return result;
}

RayModel RayModel::withBoolean(const SweptVolume& sweep,
                               BooleanOp op,
                               const vsg::dmat4& modelToWorld) const
{
    return applyBoolean(*this, sweep, op, modelToWorld);
}

} // namespace app
