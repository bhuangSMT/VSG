// RayBoolean - subtract/union a SweptVolume against independent RayGrids.
#include "RayBoolean.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include <vsg/maths/transform.h>

#include "BoundingBox.h"
#include "BVH.h"
#include "RayGrid.h"
#include "TriangleMesh.h"

namespace app
{
namespace
{

struct Hit
{
    double along = 0.0;
    Normal3f normal{0.0f, 0.0f, 0.0f};
};

struct WorldSweep
{
    const TriangleMesh& mesh;
    const BVH& bvh;
    BoundingBox worldBounds;
    vsg::dmat4 modelToWorld{};
    vsg::dmat4 worldToModel{};
};

Normal3f toNormal3f(const vsg::dvec3& n)
{
    const double len = vsg::length(n);
    if (!(len > 0.0)) return Normal3f{0.0f, 0.0f, 0.0f};
    return Normal3f{static_cast<float>(n.x / len),
                    static_cast<float>(n.y / len),
                    static_cast<float>(n.z / len)};
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

        const vsg::dvec3 e1(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        const vsg::dvec3 e2(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        const vsg::dvec3 nWorld = vsg::cross(e1, e2);
        // w=0 so translation in worldToModel does not affect the normal.
        const vsg::dvec4 nModel4 =
            sweep.worldToModel * vsg::dvec4(nWorld.x, nWorld.y, nWorld.z, 0.0);

        hits.push_back(Hit{modelHit[axis], toNormal3f(vsg::dvec3(nModel4.x, nModel4.y, nModel4.z))});
    });
}

std::vector<Interval> removalTicksFromHits(std::vector<Hit>& hits,
                                           const RayGrid& grid,
                                           double mergeTol)
{
    sortAndMerge(hits, mergeTol);

    std::vector<Interval> intervals;
    for (std::size_t i = 0; i + 1 < hits.size(); i += 2)
    {
        Interval iv;
        iv.begin = grid.toTick(hits[i].along);
        iv.end = grid.toTick(hits[i + 1].along);
        iv.beginNormal = hits[i].normal;
        iv.endNormal = hits[i + 1].normal;
        iv.setFromBoolean(true);
        if (iv.end > iv.begin) intervals.push_back(iv);
    }
    return intervals;
}

std::vector<Interval> subtractTicks(const std::vector<Interval>& solid,
                                    const std::vector<Interval>& cutters)
{
    std::vector<Interval> remaining = solid;
    for (const Interval& cut : cutters)
    {
        std::vector<Interval> next;
        for (const Interval& piece : remaining)
        {
            if (cut.end <= piece.begin || cut.begin >= piece.end)
            {
                next.push_back(piece);
                continue;
            }

            if (cut.begin > piece.begin)
            {
                Interval left = piece;
                left.end = cut.begin;
                left.endNormal = cut.beginNormal;
                if (left.end > left.begin) next.push_back(left);
            }
            if (cut.end < piece.end)
            {
                Interval right = piece;
                right.begin = cut.end;
                right.beginNormal = cut.endNormal;
                right.setFromBoolean(true);
                if (right.end > right.begin) next.push_back(right);
            }
        }
        remaining = std::move(next);
    }
    return remaining;
}

std::vector<Interval> unionTicks(const std::vector<Interval>& solid,
                                 const std::vector<Interval>& add)
{
    std::vector<Interval> all = solid;
    all.insert(all.end(), add.begin(), add.end());
    if (all.empty()) return all;

    std::sort(all.begin(), all.end(),
              [](const Interval& a, const Interval& b) { return a.begin < b.begin; });

    std::vector<Interval> merged;
    merged.push_back(all.front());
    for (std::size_t i = 1; i < all.size(); ++i)
    {
        Interval& cur = merged.back();
        const Interval& nxt = all[i];
        if (nxt.begin <= cur.end)
        {
            if (nxt.end > cur.end)
            {
                cur.end = nxt.end;
                cur.endNormal = nxt.endNormal;
            }
            if (nxt.fromBoolean()) cur.setFromBoolean(true);
        }
        else
        {
            merged.push_back(nxt);
        }
    }
    return merged;
}

void replaceSlotIntervals(RayGrid& grid, RaySlot& slot, const std::vector<Interval>& next)
{
    if (next.empty())
    {
        slot.intervalCount = 0;
        return;
    }
    if (!grid.pool.tryReplaceInPlace(slot, next.data(),
                                     static_cast<std::uint32_t>(next.size())))
        grid.pool.append(slot, next);
}

void processAxisGrid(RayGrid& grid,
                     const WorldSweep& sweep,
                     BooleanOp op,
                     double mergeTol)
{
    if (grid.empty() || !sweep.worldBounds.valid()) return;

    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;

    std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
    if (!gridWindowFromWorldAabb(grid, sweep.worldBounds, sweep.worldToModel,
                                 iu0, iu1, iv0, iv1))
        return;

    std::vector<Hit> hits;
    for (std::uint32_t iv = iv0; iv <= iv1; ++iv)
    {
        for (std::uint32_t iu = iu0; iu <= iu1; ++iu)
        {
            RaySlot& slot = grid.at(iu, iv);
            const double u0 = grid.sampleU(iu);
            const double v0 = grid.sampleV(iv);

            collectHits(sweep, axis, u, v, u0, v0, hits);
            if (hits.size() < 2)
            {
                if (op == BooleanOp::Subtraction) continue;
                // Union with empty sweep solid: unchanged.
                continue;
            }

            const std::vector<Interval> sweepSolid =
                removalTicksFromHits(hits, grid, mergeTol);
            if (sweepSolid.empty()) continue;

            std::vector<Interval> current;
            if (!slot.empty())
            {
                auto spans = grid.pool.span(slot);
                current.assign(spans.begin(), spans.end());
            }

            std::vector<Interval> result;
            if (op == BooleanOp::Subtraction)
            {
                if (current.empty()) continue;
                result = subtractTicks(current, sweepSolid);
            }
            else
            {
                result = unionTicks(current, sweepSolid);
            }

            replaceSlotIntervals(grid, slot, result);
        }
    }
}

RayGrid copyGrid(const RayGrid& src)
{
    RayGrid dst;
    dst.width = src.width;
    dst.height = src.height;
    dst.spacingU = src.spacingU;
    dst.spacingV = src.spacingV;
    dst.originU = src.originU;
    dst.originV = src.originV;
    dst.originT = src.originT;
    dst.unit = src.unit;
    dst.axis = src.axis;
    dst.cells = src.cells;
    dst.pool.data = src.pool.data;
    return dst;
}

} // namespace

BoundingBox modelAabbFromWorld(const BoundingBox& worldBounds,
                               const vsg::dmat4& worldToModel)
{
    BoundingBox out;
    if (!worldBounds.valid()) return out;

    const Point3d& wmin = worldBounds.min();
    const Point3d& wmax = worldBounds.max();
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                const vsg::dvec3 corner = worldToModel *
                    vsg::dvec3(ix ? wmax[0] : wmin[0],
                               iy ? wmax[1] : wmin[1],
                               iz ? wmax[2] : wmin[2]);
                out.expand(Point3d{corner.x, corner.y, corner.z});
            }
    return out;
}

bool gridWindowFromWorldAabb(const RayGrid& grid,
                             const BoundingBox& worldBounds,
                             const vsg::dmat4& worldToModel,
                             std::uint32_t& iu0, std::uint32_t& iu1,
                             std::uint32_t& iv0, std::uint32_t& iv1)
{
    if (grid.empty() || !worldBounds.valid()) return false;

    const BoundingBox modelAabb = modelAabbFromWorld(worldBounds, worldToModel);
    return gridWindowFromModelAabb(grid, modelAabb, iu0, iu1, iv0, iv1);
}

bool gridWindowFromModelAabb(const RayGrid& grid,
                             const BoundingBox& modelAabb,
                             std::uint32_t& iu0, std::uint32_t& iu1,
                             std::uint32_t& iv0, std::uint32_t& iv1)
{
    if (grid.empty() || !modelAabb.valid()) return false;

    const std::size_t u = (grid.axis + 1) % 3;
    const std::size_t v = (grid.axis + 2) % 3;
    const Point3d& lo = modelAabb.min();
    const Point3d& hi = modelAabb.max();

    iu0 = grid.indexU(lo[u]);
    iu1 = grid.indexU(hi[u]);
    iv0 = grid.indexV(lo[v]);
    iv1 = grid.indexV(hi[v]);
    if (iu0 > iu1) std::swap(iu0, iu1);
    if (iv0 > iv1) std::swap(iv0, iv1);
    return true;
}

RayModel applyBoolean(const RayModel& source,
                      const SweptVolume& sweep,
                      BooleanOp op,
                      const vsg::dmat4& modelToWorld)
{
    auto sourceLock = source.lockChains();

    RayModel result;
    result._bounds = source.bounds();
    result._resolution = source.resolution();

    // Deep-copy present grids so we can mutate independently.
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (const RayGrid* g = source.grid(axis))
            result._grids[axis] = copyGrid(*g);
    }

    if (op == BooleanOp::None || sweep.empty() || sweep.bvh().empty())
        return result;

    const WorldSweep worldSweep{sweep.mesh(), sweep.bvh(), sweep.bvh().bounds(),
                                modelToWorld, vsg::inverse(modelToWorld)};
    if (!worldSweep.worldBounds.valid()) return result;

    const double mergeTol = std::max(1.0e-9, worldSweep.worldBounds.diagonal() * 1.0e-9);

    // One thread per present axis; each writes only its own grid (no mutex).
    std::thread workers[2];
    int workerCount = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        RayGrid* g = result.grid(axis);
        if (!g) continue;

        if (workerCount < 2)
        {
            workers[workerCount++] = std::thread(
                [g, &worldSweep, op, mergeTol]() {
                    processAxisGrid(*g, worldSweep, op, mergeTol);
                });
        }
        else
        {
            processAxisGrid(*g, worldSweep, op, mergeTol);
        }
    }
    for (int i = 0; i < workerCount; ++i) workers[i].join();

    return result;
}

RayModel RayModel::withBoolean(const SweptVolume& sweep,
                               BooleanOp op,
                               const vsg::dmat4& modelToWorld) const
{
    return applyBoolean(*this, sweep, op, modelToWorld);
}

} // namespace app
