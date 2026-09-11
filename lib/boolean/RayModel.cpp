#include "RayModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/combinable.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "BRep.h"
#include "RayHit.h"

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

// Precomputed triangle for casting: double verts, unit normal, and per-axis
// UV bounds + barycentric coefficients (axis = cast direction).
struct CastFace
{
    struct AxisProj
    {
        double loU = 0.0, hiU = 0.0, loV = 0.0, hiV = 0.0;
        double denom = 0.0;
        double w0u = 0.0, w0v = 0.0, w0c = 0.0;
        double w1u = 0.0, w1v = 0.0, w1c = 0.0;
        bool usable = false;
    };

    Point3d a{0.0, 0.0, 0.0};
    Point3d b{0.0, 0.0, 0.0};
    Point3d c{0.0, 0.0, 0.0};
    Normal3f normal{0.0f, 0.0f, 0.0f};
    AxisProj proj[3]{};
    bool valid = false;
};

std::vector<CastFace> buildCastFaces(const BRep& brep)
{
    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();
    const std::size_t faceCount = brep.faceCount();

    std::vector<CastFace> faces(faceCount);
    for (std::size_t f = 0; f < faceCount; ++f)
    {
        const std::uint32_t begin = faceOffsets[f];
        if (faceOffsets[f + 1] - begin < 3) continue;

        CastFace& face = faces[f];
        face.a = toPoint(verts[faceVertices[begin]]);
        face.b = toPoint(verts[faceVertices[begin + 1]]);
        face.c = toPoint(verts[faceVertices[begin + 2]]);

        const double e1[3]{face.b[0] - face.a[0], face.b[1] - face.a[1], face.b[2] - face.a[2]};
        const double e2[3]{face.c[0] - face.a[0], face.c[1] - face.a[1], face.c[2] - face.a[2]};
        const double n[3]{e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};
        const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (!(length > 0.0)) continue;

        for (int i = 0; i < 3; ++i)
            face.normal[i] = static_cast<float>(n[i] / length);
        face.valid = true;

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t u = (axis + 1) % 3;
            const std::size_t v = (axis + 2) % 3;
            const double au = face.a[u], av = face.a[v];
            const double bu = face.b[u], bv = face.b[v];
            const double cu = face.c[u], cv = face.c[v];

            CastFace::AxisProj& p = face.proj[axis];
            p.loU = std::min({au, bu, cu});
            p.hiU = std::max({au, bu, cu});
            p.loV = std::min({av, bv, cv});
            p.hiV = std::max({av, bv, cv});
            p.denom = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
            const double areaScale = (p.hiU - p.loU) * (p.hiV - p.loV);
            if (std::abs(p.denom) <= 1e-12 * std::max(areaScale, 1e-300))
            {
                p.usable = false;
                continue;
            }
            const double inv = 1.0 / p.denom;
            p.w0u = (bv - cv) * inv;
            p.w0v = (cu - bu) * inv;
            p.w0c = (-(bv - cv) * cu - (cu - bu) * cv) * inv;
            p.w1u = (cv - av) * inv;
            p.w1v = (au - cu) * inv;
            p.w1c = (-(cv - av) * cu - (au - cu) * cv) * inv;
            p.usable = true;
        }
    }
    return faces;
}

void collectHits(const std::vector<CastFace>& faces,
                 const BVH& bvh,
                 std::size_t axis, std::size_t u, std::size_t v,
                 double u0, double v0,
                 std::vector<RayHit>& hits)
{
    hits.clear();

    const auto test = [&](std::size_t f) {
        if (f >= faces.size()) return;
        const CastFace& face = faces[f];
        if (!face.valid) return;

        const CastFace::AxisProj& p = face.proj[axis];
        if (!p.usable) return;
        if (!pointInUvBounds(u0, v0, p.loU, p.hiU, p.loV, p.hiV)) return;

        const double w0 = p.w0u * u0 + p.w0v * v0 + p.w0c;
        const double w1 = p.w1u * u0 + p.w1v * v0 + p.w1c;
        const double w2 = 1.0 - w0 - w1;
        if (!barycentricInside(w0, w1, w2)) return;

        RayHit hit;
        hit.along = w0 * face.a[axis] + w1 * face.b[axis] + w2 * face.c[axis];
        hit.normal = face.normal;
        hits.push_back(hit);
    };

    if (bvh.empty())
    {
        for (std::size_t f = 0; f < faces.size(); ++f) test(f);
        return;
    }

    bvh.query(u, v, u0, v0, test);
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
    const std::vector<CastFace> castFaces = buildCastFaces(brep);
    const BVH& bvh = brep.bvh();

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
        const auto castCountSz = static_cast<std::size_t>(castCount);

        std::vector<std::vector<Interval>> slotIntervals(castCountSz);
        tbb::combinable<PairingStats> pairing;

        tbb::parallel_for(
            tbb::blocked_range<int>(0, castCount),
            [&](const tbb::blocked_range<int>& range) {
                std::vector<RayHit> hits;
                hits.reserve(8);
                std::vector<Interval> local;
                local.reserve(4);
                PairingStats& localStats = pairing.local();
                for (int cast = range.begin(); cast != range.end(); ++cast)
                {
                    const std::uint32_t iu =
                        static_cast<std::uint32_t>(cast % static_cast<int>(grid.width));
                    const std::uint32_t iv =
                        static_cast<std::uint32_t>(cast / static_cast<int>(grid.width));
                    const double u0 = grid.sampleU(iu);
                    const double v0 = grid.sampleV(iv);

                    collectHits(castFaces, bvh, axis, u, v, u0, v0, hits);
                    if (hits.size() < 2) continue;

                    intervalsFromHits(hits, grid, axis, tolerance, false, local, &localStats);
                    if (!local.empty())
                    {
                        slotIntervals[static_cast<std::size_t>(cast)] = std::move(local);
                        local.clear();
                        local.reserve(4);
                    }
                }
            });

        pairing.combine_each([&](const PairingStats& s) { model._pairingStats += s; });

        std::size_t packed = 0;
        for (std::size_t i = 0; i < castCountSz; ++i)
            packed += slotIntervals[i].size();
        grid.pool.data.reserve(packed);

        for (int cast = 0; cast < castCount; ++cast)
        {
            const std::vector<Interval>& spans = slotIntervals[static_cast<std::size_t>(cast)];
            if (spans.empty()) continue;
            const std::uint32_t iu =
                static_cast<std::uint32_t>(cast % static_cast<int>(grid.width));
            const std::uint32_t iv =
                static_cast<std::uint32_t>(cast / static_cast<int>(grid.width));
            grid.pool.append(grid.at(iu, iv), spans);
        }

        model._grids[axis] = std::move(grid);
    }

    return model;
}

RayModel::RayModel(RayModel&& other) noexcept :
    _grids(std::move(other._grids)),
    _bounds(other._bounds),
    _resolution(other._resolution),
    _pairingStats(other._pairingStats)
{
}

RayModel& RayModel::operator=(RayModel&& other) noexcept
{
    if (this == &other) return *this;
    std::lock_guard<std::recursive_mutex> lock(_chainMutex);
    _grids = std::move(other._grids);
    _bounds = other._bounds;
    _resolution = other._resolution;
    _pairingStats = other._pairingStats;
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
