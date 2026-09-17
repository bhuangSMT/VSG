// RayBoolean - subtract/union a SweptVolume against independent RayGrids.
#include "RayBoolean.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include <vsg/maths/transform.h>

#include "BoundingBox.h"
#include "BVH.h"
#include "RayGrid.h"
#include "RayHit.h"
#include "TriangleMesh.h"

namespace app
{
namespace
{

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

Normal3f negateNormal(const Normal3f& n)
{
    return Normal3f{-n[0], -n[1], -n[2]};
}

void collectHits(const WorldSweep& sweep,
                 std::size_t axis, std::size_t u, std::size_t v,
                 double u0, double v0,
                 std::vector<RayHit>& hits)
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
        if (!pointInUvBounds(u0w, v0w, loU, hiU, loV, hiV)) return;

        const double denom = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
        const double areaScale = (hiU - loU) * (hiV - loV);
        if (std::abs(denom) <= 1e-12 * std::max(areaScale, 1e-300)) return;

        const double w0 = ((bv - cv) * (u0w - cu) + (cu - bu) * (v0w - cv)) / denom;
        const double w1 = ((cv - av) * (u0w - cu) + (au - cu) * (v0w - cv)) / denom;
        const double w2 = 1.0 - w0 - w1;
        if (!barycentricInside(w0, w1, w2)) return;

        const double alongWorld = w0 * a[axis] + w1 * b[axis] + w2 * c[axis];

        vsg::dvec3 worldHit;
        worldHit[u] = u0w;
        worldHit[v] = v0w;
        worldHit[axis] = alongWorld;
        const vsg::dvec3 modelHit = sweep.worldToModel * worldHit;

        // Prefer the stored outward mesh normal (set at sweep emit); fall back to
        // winding cross when the facet normal is missing / degenerate.
        vsg::dvec3 nWorld(tri.normal.x, tri.normal.y, tri.normal.z);
        if (vsg::length(nWorld) <= 1.0e-12)
        {
            const vsg::dvec3 e1(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
            const vsg::dvec3 e2(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
            nWorld = vsg::cross(e1, e2);
        }
        // w=0 so translation in worldToModel does not affect the normal.
        const vsg::dvec4 nModel4 =
            sweep.worldToModel * vsg::dvec4(nWorld.x, nWorld.y, nWorld.z, 0.0);

        hits.push_back(RayHit{modelHit[axis], toNormal3f(vsg::dvec3(nModel4.x, nModel4.y, nModel4.z))});
    });
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
                if (piece.hasSolidLength()) next.push_back(piece);
                continue;
            }

            if (cut.begin > piece.begin)
            {
                Interval left = piece;
                left.end = cut.begin;
                // Stock cavity face: opposite of tool-outward at the cutter enter.
                left.endNormal = negateNormal(cut.beginNormal);
                left.setCutEnd(true);
                if (left.hasSolidLength()) next.push_back(left);
            }
            if (cut.end < piece.end)
            {
                Interval right = piece;
                right.begin = cut.end;
                // Stock cavity face: opposite of tool-outward at the cutter leave.
                right.beginNormal = negateNormal(cut.endNormal);
                right.setFromBoolean(true);
                right.setCutBegin(true);
                if (right.hasSolidLength()) next.push_back(right);
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
                cur.setCutEnd(nxt.cutEnd());
            }
            else if (nxt.end == cur.end && nxt.cutEnd())
            {
                cur.setCutEnd(true);
                cur.endNormal = nxt.endNormal;
            }
            if (nxt.begin == cur.begin && nxt.cutBegin())
            {
                cur.setCutBegin(true);
                cur.beginNormal = nxt.beginNormal;
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

// Micro-slivers shorter than this are dropped after subtract (2–3 tick crumbs).
constexpr std::int32_t kMinKeepTicks = 4;

// Drop degenerates / micro-slivers and merge abutting or overlapping solids.
std::vector<Interval> consolidateIntervals(std::vector<Interval> spans)
{
    spans.erase(std::remove_if(spans.begin(), spans.end(),
                               [](const Interval& iv) {
                                   return (iv.end - iv.begin) < kMinKeepTicks;
                               }),
                spans.end());
    if (spans.size() < 2) return spans;

    std::sort(spans.begin(), spans.end(),
              [](const Interval& a, const Interval& b) { return a.begin < b.begin; });

    std::vector<Interval> merged;
    merged.reserve(spans.size());
    merged.push_back(spans.front());
    for (std::size_t i = 1; i < spans.size(); ++i)
    {
        Interval& cur = merged.back();
        const Interval& nxt = spans[i];
        if (nxt.begin <= cur.end)
        {
            if (nxt.end > cur.end)
            {
                cur.end = nxt.end;
                cur.endNormal = nxt.endNormal;
                cur.setCutEnd(nxt.cutEnd());
            }
            else if (nxt.end == cur.end && nxt.cutEnd())
            {
                cur.setCutEnd(true);
                cur.endNormal = nxt.endNormal;
            }
            if (nxt.begin == cur.begin && nxt.cutBegin())
            {
                cur.setCutBegin(true);
                cur.beginNormal = nxt.beginNormal;
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

// Inclusive-exclusive index range of stock intervals overlapping [tickLo, tickHi].
// Spans must be sorted by begin (cast / boolean invariant).
struct TickOverlap
{
    std::size_t begin = 0;
    std::size_t end = 0;
    bool empty() const { return begin >= end; }
};

TickOverlap overlappingTickRange(const Interval* data, std::size_t count,
                                 std::int32_t tickLo, std::int32_t tickHi)
{
    TickOverlap out;
    if (count == 0 || tickHi <= tickLo) return out;

    // First interval with end > tickLo.
    std::size_t lo = 0;
    std::size_t hi = count;
    while (lo < hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (data[mid].end <= tickLo) lo = mid + 1;
        else hi = mid;
    }
    out.begin = lo;

    // First interval with begin >= tickHi.
    lo = out.begin;
    hi = count;
    while (lo < hi)
    {
        const std::size_t mid = lo + (hi - lo) / 2;
        if (data[mid].begin < tickHi) lo = mid + 1;
        else hi = mid;
    }
    out.end = lo;
    return out;
}

void tickRangeFromModelAabb(const RayGrid& grid, const BoundingBox& modelAabb,
                            std::int32_t& tickLo, std::int32_t& tickHi)
{
    const Point3d& lo = modelAabb.min();
    const Point3d& hi = modelAabb.max();
    tickLo = grid.toTick(lo[grid.axis]) - 1;
    tickHi = grid.toTick(hi[grid.axis]) + 1;
    if (tickLo > tickHi) std::swap(tickLo, tickHi);
}

void maybeCompactPool(RayGrid& grid)
{
    // O(1): append() maintains the orphan count, so a cut that touched a small
    // window does not pay a scan over every slot in the grid.
    IntervalPool& pool = grid.pool;
    if (pool.wasted == 0) return;
    if (pool.wasted * 2 > pool.data.size()) pool.compact(grid.cells);
}

// Keep the existing (origin, spacing) lattice and add rows/columns so modelAabb
// is covered. Existing slots keep their intervals; new slots start empty.
void growGridToCover(RayGrid& grid, const BoundingBox& modelAabb)
{
    if (grid.empty() || !modelAabb.valid()) return;
    if (!(grid.spacingU > 0.0f) || !(grid.spacingV > 0.0f)) return;

    const auto u = (grid.axis + 1) % 3;
    const auto v = (grid.axis + 2) % 3;
    const Point3d& lo = modelAabb.min();
    const Point3d& hi = modelAabb.max();

    const double spacingU = static_cast<double>(grid.spacingU);
    const double spacingV = static_cast<double>(grid.spacingV);
    double originU = static_cast<double>(grid.originU);
    double originV = static_cast<double>(grid.originV);

    auto padCount = [](double origin, double spacing, double needMin) {
        if (!(needMin < origin)) return 0;
        const double raw = (origin - needMin) / spacing;
        int pad = static_cast<int>(std::ceil(raw - 1.0e-12));
        if (pad < 1) pad = 1;
        while (origin - static_cast<double>(pad) * spacing > needMin + 1.0e-12)
            ++pad;
        return pad;
    };

    const int padLeft = padCount(originU, spacingU, lo[u]);
    const int padBottom = padCount(originV, spacingV, lo[v]);
    originU -= static_cast<double>(padLeft) * spacingU;
    originV -= static_cast<double>(padBottom) * spacingV;

    auto lastIndex = [](double origin, double spacing, double needMax) {
        const double raw = (needMax - origin) / spacing;
        long long i = static_cast<long long>(std::floor(raw));
        if (i < 0) i = 0;
        return i;
    };

    constexpr int maxDim = 8193;
    long long newLastU = lastIndex(originU, spacingU, hi[u]);
    long long newLastV = lastIndex(originV, spacingV, hi[v]);
    const long long oldLastU = static_cast<long long>(grid.width) - 1 + padLeft;
    const long long oldLastV = static_cast<long long>(grid.height) - 1 + padBottom;
    if (newLastU < oldLastU) newLastU = oldLastU;
    if (newLastV < oldLastV) newLastV = oldLastV;

    long long newWidth = newLastU + 1;
    long long newHeight = newLastV + 1;
    if (newWidth > maxDim) newWidth = maxDim;
    if (newHeight > maxDim) newHeight = maxDim;

    const long long curWidth = static_cast<long long>(grid.width);
    const long long curHeight = static_cast<long long>(grid.height);
    if (padLeft == 0 && padBottom == 0 && newWidth == curWidth && newHeight == curHeight)
        return;

    const auto grownW = static_cast<std::uint32_t>(newWidth);
    const auto grownH = static_cast<std::uint32_t>(newHeight);
    std::vector<RaySlot> grown;
    grown.assign(grownW * grownH, RaySlot{});

    const auto padU = static_cast<std::uint32_t>(padLeft);
    const auto padV = static_cast<std::uint32_t>(padBottom);
    for (std::uint32_t iv = 0; iv < grid.height; ++iv)
    {
        const std::uint32_t destV = iv + padV;
        if (destV >= grownH) continue;
        for (std::uint32_t iu = 0; iu < grid.width; ++iu)
        {
            const std::uint32_t destU = iu + padU;
            if (destU >= grownW) continue;
            grown[destV * grownW + destU] = grid.cells[iv * grid.width + iu];
        }
    }

    grid.cells.swap(grown);
    grid.width = grownW;
    grid.height = grownH;
    grid.originU = static_cast<float>(originU);
    grid.originV = static_cast<float>(originV);
}

// Cells whose intervals changed. Pool writes stay on one thread after the
// parallel window walk; workers only read.
struct PendingUpdate
{
    std::uint32_t iu = 0;
    std::uint32_t iv = 0;
    std::vector<Interval> intervals; // empty => clear the slot
};

struct CellScratch
{
    std::vector<RayHit> hits;
    std::vector<Interval> current;
    std::vector<PendingUpdate> pending;
    PairingStats pairing;
};

struct CellUv
{
    std::uint32_t iu = 0;
    std::uint32_t iv = 0;
};

// Mark grid cells covered by any sweep triangle's lateral UV box (model space),
// clipped to clipRegion. Returns sorted unique (iu,iv) list.
std::vector<CellUv> markCellsFromTriangles(const RayGrid& grid,
                                           const WorldSweep& sweep,
                                           const BoundingBox& clipRegion,
                                           std::uint32_t iu0, std::uint32_t iu1,
                                           std::uint32_t iv0, std::uint32_t iv1)
{
    std::vector<CellUv> out;
    if (grid.empty() || !clipRegion.valid() || iu0 > iu1 || iv0 > iv1) return out;

    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const Point3d& clo = clipRegion.min();
    const Point3d& chi = clipRegion.max();

    const std::uint32_t windowW = iu1 - iu0 + 1;
    const std::uint32_t windowH = iv1 - iv0 + 1;
    const std::size_t cellCount =
        static_cast<std::size_t>(windowW) * static_cast<std::size_t>(windowH);
    std::vector<std::uint8_t> marked(cellCount, 0);

    for (const MeshTriangle& tri : sweep.mesh.triangles)
    {
        const vsg::dvec3 a =
            sweep.worldToModel * vsg::dvec3(tri.v0.x, tri.v0.y, tri.v0.z);
        const vsg::dvec3 b =
            sweep.worldToModel * vsg::dvec3(tri.v1.x, tri.v1.y, tri.v1.z);
        const vsg::dvec3 c =
            sweep.worldToModel * vsg::dvec3(tri.v2.x, tri.v2.y, tri.v2.z);

        double loU = std::min({a[u], b[u], c[u]});
        double hiU = std::max({a[u], b[u], c[u]});
        double loV = std::min({a[v], b[v], c[v]});
        double hiV = std::max({a[v], b[v], c[v]});

        loU = std::max(loU, clo[u]);
        hiU = std::min(hiU, chi[u]);
        loV = std::max(loV, clo[v]);
        hiV = std::min(hiV, chi[v]);
        if (loU > hiU || loV > hiV) continue;

        std::uint32_t tu0 = grid.indexU(loU);
        std::uint32_t tu1 = grid.indexU(hiU);
        std::uint32_t tv0 = grid.indexV(loV);
        std::uint32_t tv1 = grid.indexV(hiV);
        if (tu0 > tu1) std::swap(tu0, tu1);
        if (tv0 > tv1) std::swap(tv0, tv1);
        tu0 = std::max(tu0, iu0);
        tu1 = std::min(tu1, iu1);
        tv0 = std::max(tv0, iv0);
        tv1 = std::min(tv1, iv1);
        if (tu0 > tu1 || tv0 > tv1) continue;

        for (std::uint32_t iv = tv0; iv <= tv1; ++iv)
        {
            const std::size_t row =
                static_cast<std::size_t>(iv - iv0) * static_cast<std::size_t>(windowW);
            for (std::uint32_t iu = tu0; iu <= tu1; ++iu)
                marked[row + static_cast<std::size_t>(iu - iu0)] = 1;
        }
    }

    out.reserve(cellCount / 8 + 1);
    for (std::uint32_t iv = iv0; iv <= iv1; ++iv)
    {
        const std::size_t row =
            static_cast<std::size_t>(iv - iv0) * static_cast<std::size_t>(windowW);
        for (std::uint32_t iu = iu0; iu <= iu1; ++iu)
        {
            if (marked[row + static_cast<std::size_t>(iu - iu0)])
                out.push_back(CellUv{iu, iv});
        }
    }
    return out;
}

void processAxisGrid(RayGrid& grid,
                     const WorldSweep& sweep,
                     BooleanOp op,
                     double mergeTol,
                     const BoundingBox& stockBounds,
                     PairingStats& pairing,
                     std::size_t& dirtyCells)
{
    if (grid.empty() || !sweep.worldBounds.valid()) return;

    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;

    const BoundingBox sweepModelAabb =
        modelAabbFromWorld(sweep.worldBounds, sweep.worldToModel);
    if (!sweepModelAabb.valid()) return;

    // Subtraction: only stock∩sweep. Union: allow coverage past stock for new cells.
    BoundingBox clipRegion = stockBounds;
    BoundingBox windowAabb = sweepModelAabb;
    if (op == BooleanOp::Union)
    {
        clipRegion.expand(sweepModelAabb);
    }
    else
    {
        windowAabb = sweepModelAabb.intersection(stockBounds);
        if (!windowAabb.valid()) return;
        clipRegion = stockBounds;
    }

    std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
    if (!gridWindowFromModelAabb(grid, windowAabb, iu0, iu1, iv0, iv1))
        return;

    std::int32_t tickLo = 0;
    std::int32_t tickHi = 0;
    tickRangeFromModelAabb(grid, windowAabb, tickLo, tickHi);

    const std::vector<CellUv> cells =
        markCellsFromTriangles(grid, sweep, clipRegion, iu0, iu1, iv0, iv1);
    dirtyCells += cells.size();
    if (cells.empty()) return;

    tbb::enumerable_thread_specific<CellScratch> scratch;

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, cells.size()),
        [&](const tbb::blocked_range<std::size_t>& range) {
            CellScratch& local = scratch.local();

            for (std::size_t idx = range.begin(); idx != range.end(); ++idx)
            {
                const std::uint32_t iu = cells[idx].iu;
                const std::uint32_t iv = cells[idx].iv;

                const RaySlot& slot = grid.at(iu, iv);
                const double u0 = grid.sampleU(iu);
                const double v0 = grid.sampleV(iv);

                IntervalSpan spans;
                TickOverlap overlap;
                if (!slot.empty())
                {
                    spans = grid.pool.span(slot);
                    overlap = overlappingTickRange(spans.data(), spans.size(), tickLo, tickHi);
                }

                // Subtract: skip rays with no stock in the current-move tick band
                // before paying for BVH triangle tests.
                if (op == BooleanOp::Subtraction && overlap.empty()) continue;

                collectHits(sweep, axis, u, v, u0, v0, local.hits);
                if (local.hits.size() < 2) continue;

                std::vector<Interval> sweepSolid;
                intervalsFromHits(local.hits, grid, axis, mergeTol, true, sweepSolid,
                                  &local.pairing);
                if (sweepSolid.empty()) continue;

                std::vector<Interval> result;
                if (op == BooleanOp::Subtraction)
                {
                    local.current.assign(spans.begin() + static_cast<std::ptrdiff_t>(overlap.begin),
                                         spans.begin() + static_cast<std::ptrdiff_t>(overlap.end));
                    std::vector<Interval> mid =
                        consolidateIntervals(subtractTicks(local.current, sweepSolid));

                    result.reserve((overlap.begin) + mid.size() + (spans.size() - overlap.end));
                    result.insert(result.end(), spans.begin(),
                                  spans.begin() + static_cast<std::ptrdiff_t>(overlap.begin));
                    result.insert(result.end(), mid.begin(), mid.end());
                    result.insert(result.end(),
                                  spans.begin() + static_cast<std::ptrdiff_t>(overlap.end),
                                  spans.end());
                    result = consolidateIntervals(std::move(result));
                }
                else
                {
                    local.current.clear();
                    if (!overlap.empty())
                    {
                        local.current.assign(
                            spans.begin() + static_cast<std::ptrdiff_t>(overlap.begin),
                            spans.begin() + static_cast<std::ptrdiff_t>(overlap.end));
                    }
                    std::vector<Interval> mid =
                        consolidateIntervals(unionTicks(local.current, sweepSolid));

                    result.reserve(overlap.begin + mid.size() + (spans.size() - overlap.end));
                    if (!spans.empty())
                    {
                        result.insert(result.end(), spans.begin(),
                                      spans.begin() + static_cast<std::ptrdiff_t>(overlap.begin));
                    }
                    result.insert(result.end(), mid.begin(), mid.end());
                    if (!spans.empty())
                    {
                        result.insert(result.end(),
                                      spans.begin() + static_cast<std::ptrdiff_t>(overlap.end),
                                      spans.end());
                    }
                    result = consolidateIntervals(std::move(result));
                }

                local.pending.push_back(PendingUpdate{iu, iv, std::move(result)});
            }
        });

    std::vector<PendingUpdate> pending;
    for (CellScratch& local : scratch)
    {
        pending.insert(pending.end(), std::make_move_iterator(local.pending.begin()),
                       std::make_move_iterator(local.pending.end()));
        pairing += local.pairing;
    }
    std::sort(pending.begin(), pending.end(),
              [](const PendingUpdate& lhs, const PendingUpdate& rhs) {
                  if (lhs.iv != rhs.iv) return lhs.iv < rhs.iv;
                  return lhs.iu < rhs.iu;
              });

    for (const PendingUpdate& entry : pending)
    {
        RaySlot& slot = grid.at(entry.iu, entry.iv);
        if (entry.intervals.empty())
        {
            slot.intervalCount = 0;
            continue;
        }
        if (!grid.pool.tryReplaceInPlace(slot, entry.intervals.data(),
                                         static_cast<std::uint32_t>(entry.intervals.size())))
            grid.pool.append(slot, entry.intervals);
    }

    maybeCompactPool(grid);
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
    dst.pool.wasted = src.pool.wasted;
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

RayModel RayModel::clone() const
{
    auto lock = lockChains();

    RayModel result;
    result._bounds = _bounds;
    result._resolution = _resolution;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (const RayGrid* g = grid(axis))
            result._grids[axis] = copyGrid(*g);
    }
    return result;
}

RayModel applyBoolean(const RayModel& source,
                      const SweptVolume& sweep,
                      BooleanOp op,
                      const vsg::dmat4& modelToWorld)
{
    RayModel result = source.clone();
    applyBooleanInPlace(result, sweep, op, modelToWorld);
    return result;
}

void applyBooleanInPlace(RayModel& model,
                         const SweptVolume& sweep,
                         BooleanOp op,
                         const vsg::dmat4& modelToWorld)
{
    auto lock = model.lockChains();

    if (op == BooleanOp::None || op == BooleanOp::Probe || sweep.empty() || sweep.bvh().empty())
        return;
    if (op == BooleanOp::Inspection)
        op = BooleanOp::Subtraction;

    const WorldSweep worldSweep{sweep.mesh(), sweep.bvh(), sweep.bvh().bounds(),
                                modelToWorld, vsg::inverse(modelToWorld)};
    if (!worldSweep.worldBounds.valid()) return;

    const double mergeTol = std::max(1.0e-9, worldSweep.worldBounds.diagonal() * 1.0e-9);

    model._pairingStats = {};
    model._lastDirtyCellCount = 0;

    const BoundingBox sweepModelAabb =
        modelAabbFromWorld(worldSweep.worldBounds, worldSweep.worldToModel);

    // Union past the stock AABB: keep the lattice and append the extra rows
    // and columns. Without this, existing X/Y rays lengthen into the sweep
    // and the corner between them is never sampled.
    if (op == BooleanOp::Union && sweepModelAabb.valid())
    {
        BoundingBox coverage = model._bounds;
        coverage.expand(sweepModelAabb);
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (RayGrid* g = model.grid(axis))
                growGridToCover(*g, coverage);
        }
    }

    // Axes share nothing, but nested TBB (axis × cell) raced the interval
    // pool. Walk axes in order; each axis still parallelizes its dirty cells.
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (RayGrid* g = model.grid(axis))
            processAxisGrid(*g, worldSweep, op, mergeTol, model._bounds,
                            model._pairingStats, model._lastDirtyCellCount);
    }
}

RayModel RayModel::withBoolean(const SweptVolume& sweep,
                               BooleanOp op,
                               const vsg::dmat4& modelToWorld) const
{
    return applyBoolean(*this, sweep, op, modelToWorld);
}

void RayModel::booleanInPlace(const SweptVolume& sweep,
                              BooleanOp op,
                              const vsg::dmat4& modelToWorld)
{
    applyBooleanInPlace(*this, sweep, op, modelToWorld);
}

void RayModel::shellInPlace(double thickness)
{
    if (!(thickness > 0.0))
        throw std::invalid_argument("Shell thickness must be positive.");

    auto lock = lockChains();

    auto negate = [](const Normal3f& n) {
        return Normal3f{-n[0], -n[1], -n[2]};
    };
    // Reversed outward normal for a new offset end. Falls back to the axis
    // direction opposite the original enter/exit convention when the stored
    // normal is degenerate (overlay / GS then still have a sheet direction).
    auto reversedFrom = [&](const Normal3f& n, std::size_t axis, bool originalEnter) {
        const float len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        if (len2 < 1.0e-12f)
        {
            Normal3f out{0.0f, 0.0f, 0.0f};
            out[axis] = originalEnter ? 1.0f : -1.0f;
            return out;
        }
        return negate(n);
    };

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        RayGrid* g = grid(axis);
        if (!g || g->empty() || !(g->unit > 0.0f)) continue;

        const double ticks = thickness / static_cast<double>(g->unit);
        std::int32_t offset = static_cast<std::int32_t>(std::llround(ticks));
        if (offset < 1) offset = 1;

        for (RaySlot& slot : g->cells)
        {
            if (slot.intervalCount == 0) continue;

            const IntervalSpan span = g->pool.span(slot);
            std::vector<Interval> walls;
            walls.reserve(static_cast<std::size_t>(span.count) * 2);

            for (const Interval* it = span.begin(); it != span.end(); ++it)
            {
                const std::int32_t begin = it->begin;
                const std::int32_t end = it->end;
                const std::int32_t innerEnd = begin + offset;
                const std::int32_t innerBegin = end - offset;

                // Overlapping walls from the same parent: drop both.
                if (innerEnd > innerBegin) continue;

                Interval entryWall;
                entryWall.begin = begin;
                entryWall.end = innerEnd;
                entryWall.beginNormal = it->beginNormal;
                // New offset end: reverse of the original entry normal.
                entryWall.endNormal = reversedFrom(it->beginNormal, axis, true);
                entryWall.flags = 0;
                if (it->fromBoolean()) entryWall.setFromBoolean(true);
                // Splat color follows cut flags (tool vs stock). Match the new
                // offset end to the paired original entry.
                if (it->cutBegin())
                {
                    entryWall.setCutBegin(true);
                    entryWall.setCutEnd(true);
                }

                Interval exitWall;
                exitWall.begin = innerBegin;
                exitWall.end = end;
                // New offset begin: reverse of the original exit normal.
                exitWall.beginNormal = reversedFrom(it->endNormal, axis, false);
                exitWall.endNormal = it->endNormal;
                exitWall.flags = 0;
                if (it->fromBoolean()) exitWall.setFromBoolean(true);
                if (it->cutEnd())
                {
                    exitWall.setCutEnd(true);
                    exitWall.setCutBegin(true);
                }

                if (entryWall.hasSolidLength()) walls.push_back(entryWall);
                if (exitWall.hasSolidLength()) walls.push_back(exitWall);
            }

            if (walls.empty())
            {
                g->pool.wasted += slot.capacity();
                slot.intervalCount = 0;
                slot.intervalCapacity = 0;
                slot.intervalOffset = 0;
                continue;
            }

            if (!g->pool.tryReplaceInPlace(slot, walls.data(),
                                           static_cast<std::uint32_t>(walls.size())))
                g->pool.append(slot, walls);
        }

        g->pool.compact(g->cells);
    }
}

} // namespace app
