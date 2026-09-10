#include "RayModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "BRep.h"

namespace app
{

namespace
{

constexpr double maxStepsPerAxis = 8192.0;
constexpr double maxCastsPerDirection = 8.0e6;

struct CastExtent
{
    double uSteps = 0.0;
    double vSteps = 0.0;
    double casts = 0.0;
};

CastExtent castExtent(const BoundingBox& bounds, const Point3d& resolution, std::size_t axis)
{
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;

    CastExtent extent;
    extent.uSteps = std::floor(bounds.extent(u) / resolution[u]);
    extent.vSteps = std::floor(bounds.extent(v) / resolution[v]);
    extent.casts = (extent.uSteps + 1.0) * (extent.vSteps + 1.0);
    return extent;
}

bool withinCastBudget(const BoundingBox& bounds, const Point3d& resolution)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const CastExtent extent = castExtent(bounds, resolution, axis);
        if (extent.uSteps > maxStepsPerAxis || extent.vSteps > maxStepsPerAxis ||
            extent.casts > maxCastsPerDirection)
            return false;
    }
    return true;
}

Point3d toPoint(const vsg::vec3& v)
{
    return Point3d{static_cast<double>(v.x),
                   static_cast<double>(v.y),
                   static_cast<double>(v.z)};
}

struct Hit
{
    double along = 0.0;
    Normal3f normal{0.0f, 0.0f, 0.0f};
};

void collectHits(const BRep& brep,
                 std::size_t axis, std::size_t u, std::size_t v,
                 double u0, double v0,
                 std::vector<Hit>& hits)
{
    hits.clear();

    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();

    const auto test = [&](std::size_t f) {
        const std::uint32_t begin = faceOffsets[f];
        if (faceOffsets[f + 1] - begin < 3) return;

        const Point3d a = toPoint(verts[faceVertices[begin]]);
        const Point3d b = toPoint(verts[faceVertices[begin + 1]]);
        const Point3d c = toPoint(verts[faceVertices[begin + 2]]);

        const double au = a[u], av = a[v];
        const double bu = b[u], bv = b[v];
        const double cu = c[u], cv = c[v];

        const double loU = std::min({au, bu, cu});
        const double hiU = std::max({au, bu, cu});
        const double loV = std::min({av, bv, cv});
        const double hiV = std::max({av, bv, cv});
        if (u0 < loU || u0 > hiU || v0 < loV || v0 > hiV) return;

        const double denom = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
        const double areaScale = (hiU - loU) * (hiV - loV);
        if (std::abs(denom) <= 1e-12 * std::max(areaScale, 1e-300)) return;

        const double w0 = ((bv - cv) * (u0 - cu) + (cu - bu) * (v0 - cv)) / denom;
        const double w1 = ((cv - av) * (u0 - cu) + (au - cu) * (v0 - cv)) / denom;
        const double w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) return;

        Hit hit;
        hit.along = w0 * a[axis] + w1 * b[axis] + w2 * c[axis];

        const double e1[3]{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3]{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double n[3]{e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};

        const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (length > 0.0)
        {
            for (int i = 0; i < 3; ++i)
                hit.normal[i] = static_cast<float>(n[i] / length);
        }

        hits.push_back(hit);
    };

    const BVH& bvh = brep.bvh();
    if (bvh.empty())
    {
        const std::size_t faces = brep.faceCount();
        for (std::size_t f = 0; f < faces; ++f) test(f);
        return;
    }

    bvh.query(u, v, u0, v0, test);
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

Point3d finestWithin(const BoundingBox& bounds, const Point3d& requested)
{
    if (!bounds.valid() || withinCastBudget(bounds, requested)) return requested;

    double scale = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const CastExtent extent = castExtent(bounds, requested, axis);

        if (extent.casts > maxCastsPerDirection)
            scale = std::max(scale, std::sqrt(extent.casts / maxCastsPerDirection));

        scale = std::max(scale, extent.uSteps / maxStepsPerAxis);
        scale = std::max(scale, extent.vSteps / maxStepsPerAxis);
    }

    Point3d resolution = requested;
    for (int attempt = 0; attempt < 128; ++attempt)
    {
        for (std::size_t i = 0; i < 3; ++i) resolution[i] = requested[i] * scale;

        if (withinCastBudget(bounds, resolution)) break;
        scale *= 1.02;
    }

    return resolution;
}

RayGrid buildGridFromBounds(const BoundingBox& bounds,
                            const Point3d& resolution,
                            std::size_t axis)
{
    RayGrid grid;
    grid.axis = axis;

    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const Point3d& lo = bounds.min();

    const CastExtent extent = castExtent(bounds, resolution, axis);
    const int uCount = (extent.uSteps > 0.0) ? static_cast<int>(extent.uSteps) : 0;
    const int vCount = (extent.vSteps > 0.0) ? static_cast<int>(extent.vSteps) : 0;

    grid.width = static_cast<std::uint32_t>(uCount + 1);
    grid.height = static_cast<std::uint32_t>(vCount + 1);
    grid.spacingU = static_cast<float>(resolution[u]);
    grid.spacingV = static_cast<float>(resolution[v]);
    grid.originU = static_cast<float>(lo[u]);
    grid.originV = static_cast<float>(lo[v]);
    grid.originT = static_cast<float>(lo[axis]);
    grid.unit = static_cast<float>(resolution[axis]);
    if (!(grid.unit > 0.0f)) grid.unit = 1.0f;

    grid.cells.assign(static_cast<std::size_t>(grid.width) * static_cast<std::size_t>(grid.height),
                      RaySlot{});
    return grid;
}

} // namespace

Point3d RayModel::finestCastableResolution(const BRep& brep, const Point3d& requested)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (!(requested[i] > 0.0))
            throw std::invalid_argument("RayModel resolution must be positive along every axis.");
    }

    return finestWithin(BoundingBox::fromBRep(brep), requested);
}

RayModel RayModel::fromBRep(const BRep& brep, const Point3d& resolution)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (!(resolution[i] > 0.0))
            throw std::invalid_argument("RayModel resolution must be positive along every axis.");
    }

    RayModel model;
    model._resolution = resolution;
    model._bounds = BoundingBox::fromBRep(brep);

    if (!model._bounds.valid() || brep.faceCount() == 0) return model;

    const double tolerance = 1e-9 * std::max(model._bounds.diagonal(), 1.0);

    if (!withinCastBudget(model._bounds, resolution))
    {
        const Point3d finest = finestWithin(model._bounds, resolution);
        const CastExtent extent = castExtent(model._bounds, resolution, 0);
        throw std::invalid_argument(
            "The ray resolution is too fine to cast for this model: it would need " +
            std::to_string(static_cast<long long>(extent.casts)) +
            " casts in one direction, and the limit is " +
            std::to_string(static_cast<long long>(maxCastsPerDirection)) +
            ". The finest that will cast is " + std::to_string(finest[0]) + ", " +
            std::to_string(finest[1]) + ", " + std::to_string(finest[2]) + ".");
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const std::size_t u = (axis + 1) % 3;
        const std::size_t v = (axis + 2) % 3;

        RayGrid grid = buildGridFromBounds(model._bounds, resolution, axis);
        const int castCount = static_cast<int>(grid.width * grid.height);

        // Per-slot interval lists built in parallel, then packed into the pool.
        std::vector<std::vector<Interval>> slotIntervals(
            static_cast<std::size_t>(castCount));

        tbb::parallel_for(
            tbb::blocked_range<int>(0, castCount),
            [&](const tbb::blocked_range<int>& range) {
                std::vector<Hit> hits;
                for (int cast = range.begin(); cast != range.end(); ++cast)
                {
                    const std::uint32_t iu = static_cast<std::uint32_t>(cast % static_cast<int>(grid.width));
                    const std::uint32_t iv = static_cast<std::uint32_t>(cast / static_cast<int>(grid.width));
                    const double u0 = grid.sampleU(iu);
                    const double v0 = grid.sampleV(iv);

                    collectHits(brep, axis, u, v, u0, v0, hits);
                    if (hits.size() < 2) continue;

                    sortAndMerge(hits, tolerance);

                    std::vector<Interval>& out = slotIntervals[static_cast<std::size_t>(cast)];
                    out.reserve(hits.size() / 2);
                    for (std::size_t h = 0; h + 1 < hits.size(); h += 2)
                    {
                        Interval ivSpan;
                        ivSpan.begin = grid.toTick(hits[h].along);
                        ivSpan.end = grid.toTick(hits[h + 1].along);
                        ivSpan.beginNormal = hits[h].normal;
                        ivSpan.endNormal = hits[h + 1].normal;
                        if (ivSpan.end > ivSpan.begin) out.push_back(ivSpan);
                    }
                }
            });

        for (int cast = 0; cast < castCount; ++cast)
        {
            const std::vector<Interval>& spans = slotIntervals[static_cast<std::size_t>(cast)];
            if (spans.empty()) continue;
            const std::uint32_t iu = static_cast<std::uint32_t>(cast % static_cast<int>(grid.width));
            const std::uint32_t iv = static_cast<std::uint32_t>(cast / static_cast<int>(grid.width));
            grid.pool.append(grid.at(iu, iv), spans);
        }

        model._grids[axis] = std::move(grid);
    }

    return model;
}

RayModel::RayModel(RayModel&& other) noexcept :
    _grids(std::move(other._grids)),
    _bounds(other._bounds),
    _resolution(other._resolution)
{
}

RayModel& RayModel::operator=(RayModel&& other) noexcept
{
    if (this == &other) return *this;
    std::lock_guard<std::recursive_mutex> lock(_chainMutex);
    _grids = std::move(other._grids);
    _bounds = other._bounds;
    _resolution = other._resolution;
    return *this;
}

const RayGrid* RayModel::grid(std::size_t axis) const
{
    if (axis >= 3 || !_grids[axis]) return nullptr;
    return &*_grids[axis];
}

RayGrid* RayModel::grid(std::size_t axis)
{
    if (axis >= 3 || !_grids[axis]) return nullptr;
    return &*_grids[axis];
}

std::unique_lock<std::recursive_mutex> RayModel::lockChains() const
{
    return std::unique_lock<std::recursive_mutex>(_chainMutex);
}

std::size_t RayModel::rayCount() const
{
    auto lock = lockChains();
    std::size_t total = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (const RayGrid* g = grid(axis)) total += g->intervalCount();
    }
    return total;
}

int RayModel::strideForRayBudget(std::size_t maxRays) const
{
    if (maxRays == 0) return 1;

    const std::size_t total = rayCount();
    if (total <= maxRays) return 1;

    const double estimate = std::sqrt(static_cast<double>(total) / static_cast<double>(maxRays));
    int stride = (estimate > 1.0) ? static_cast<int>(estimate) : 1;
    while (stride < maxStride && rayCountAtStride(stride) > maxRays) ++stride;
    return stride;
}

std::size_t RayModel::rayCountAtStride(int stride) const
{
    if (stride <= 1) return rayCount();

    auto lock = lockChains();
    std::size_t total = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (const RayGrid* g = grid(axis)) total += g->intervalCountAtStride(stride);
    }
    return total;
}

} // namespace app
