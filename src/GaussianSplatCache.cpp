// GaussianSplatCache - fixed-capacity per-cell GPU splat buffers for Ray-GS.
#include "GaussianSplatCache.h"

#include <cmath>
#include <stdexcept>

#include "RayBoolean.h"
#include "RayGrid.h"

namespace app
{
namespace
{

vsg::vec3 normalOrAxis(const Normal3f& n, std::size_t axis, bool enter)
{
    const float len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (len2 < 1.0e-12f)
    {
        vsg::vec3 fallback(0.0f, 0.0f, 0.0f);
        fallback[static_cast<int>(axis)] = enter ? -1.0f : 1.0f;
        return fallback;
    }
    return vsg::vec3(n[0], n[1], n[2]);
}

std::uint32_t sampledCount(std::uint32_t extent, int stride)
{
    if (extent == 0 || stride <= 0) return 0;
    return (extent + static_cast<std::uint32_t>(stride) - 1) /
           static_cast<std::uint32_t>(stride);
}

std::uint32_t sampledIndex(std::uint32_t i, int stride)
{
    return i / static_cast<std::uint32_t>(stride);
}

} // namespace

void GaussianSplatCache::clear()
{
    _set.resize(0);
    _axes = {};
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _capacity = 0;
}

bool GaussianSplatCache::layoutMatches(const RayModel& rayModel, int stride) const
{
    if (stride != _stride || rayModel.resolution() != _resolution) return false;
    if (_capacity == 0 || _set.empty()) return false;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        const AxisLayout& layout = _axes[axis];
        if (!grid)
        {
            if (layout.present) return false;
            continue;
        }
        if (!layout.present) return false;
        if (layout.width != grid->width || layout.height != grid->height) return false;
        if (layout.sampledW != sampledCount(grid->width, stride) ||
            layout.sampledH != sampledCount(grid->height, stride))
            return false;
    }
    return true;
}

void GaussianSplatCache::writeCell(const RayModel& rayModel,
                                   std::size_t axis,
                                   std::uint32_t iu,
                                   std::uint32_t iv,
                                   float radius,
                                   const SplatStyle& style,
                                   bool* overflow)
{
    const AxisLayout& layout = _axes[axis];
    if (!layout.present) return;

    const RayGrid* grid = rayModel.grid(axis);
    if (!grid) return;

    const std::uint32_t su = sampledIndex(iu, _stride);
    const std::uint32_t sv = sampledIndex(iv, _stride);
    if (su >= layout.sampledW || sv >= layout.sampledH) return;

    const std::size_t cellBase =
        layout.baseSplat +
        (static_cast<std::size_t>(sv) * layout.sampledW + su) *
            static_cast<std::size_t>(maxEndpointsPerCell);

    const RaySlot& slot = grid->at(iu, iv);
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid->sampleU(iu);
    const double v0 = grid->sampleV(iv);

    vsg::vec4 stock = style.stockColor;
    stock.a = style.opacity;
    vsg::vec4 tool = style.toolColor;
    tool.a = style.opacity;

    std::size_t written = 0;
    if (!slot.empty())
    {
        auto spans = grid->pool.span(slot);
        for (const Interval& span : spans)
        {
            if (written + 2 > static_cast<std::size_t>(maxEndpointsPerCell))
            {
                if (overflow) *overflow = true;
                break;
            }

            Point3d start{0.0, 0.0, 0.0};
            Point3d end{0.0, 0.0, 0.0};
            start[axis] = grid->fromTick(span.begin);
            end[axis] = grid->fromTick(span.end);
            start[u] = u0;
            end[u] = u0;
            start[v] = v0;
            end[v] = v0;

            const vsg::vec4& color = span.fromBoolean() ? tool : stock;
            _set.set(cellBase + written,
                     {vsg::vec3(static_cast<float>(start[0]),
                                static_cast<float>(start[1]),
                                static_cast<float>(start[2])),
                      normalOrAxis(span.beginNormal, axis, true), color, radius});
            _set.set(cellBase + written + 1,
                     {vsg::vec3(static_cast<float>(end[0]),
                                static_cast<float>(end[1]),
                                static_cast<float>(end[2])),
                      normalOrAxis(span.endNormal, axis, false), color, radius});
            written += 2;
        }
    }

    for (; written < static_cast<std::size_t>(maxEndpointsPerCell); ++written)
        _set.clearSlot(cellBase + written);
}

vsg::ref_ptr<vsg::Node> GaussianSplatCache::rebuild(const RayModel& rayModel,
                                                    int stride,
                                                    const std::array<float, 3>& radii,
                                                    const SplatStyle& style)
{
    auto lock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    if (stride < 1) stride = 1;

    _axes = {};
    _resolution = rayModel.resolution();
    _stride = stride;

    std::size_t capacity = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;

        AxisLayout& layout = _axes[axis];
        layout.present = true;
        layout.width = grid->width;
        layout.height = grid->height;
        layout.sampledW = sampledCount(grid->width, stride);
        layout.sampledH = sampledCount(grid->height, stride);
        layout.baseSplat = capacity;
        capacity += static_cast<std::size_t>(layout.sampledW) *
                    static_cast<std::size_t>(layout.sampledH) *
                    static_cast<std::size_t>(maxEndpointsPerCell);
    }

    _capacity = capacity;
    _set.resize(capacity);

    bool overflow = false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const RayGrid* grid = rayModel.grid(axis);
        const float radius = radii[axis];

        for (std::uint32_t iv = 0; iv < grid->height; ++iv)
        {
            if (static_cast<int>(iv) % stride != 0) continue;
            for (std::uint32_t iu = 0; iu < grid->width; ++iu)
            {
                if (static_cast<int>(iu) % stride != 0) continue;
                writeCell(rayModel, axis, iu, iv, radius, style, &overflow);
            }
        }
    }

    _set.markDirty();
    return _set.node();
}

bool GaussianSplatCache::updateRegion(const RayModel& rayModel,
                                      const BoundingBox& modelAabb,
                                      int stride,
                                      const std::array<float, 3>& radii,
                                      const SplatStyle& style)
{
    if (!layoutMatches(rayModel, stride)) return false;
    if (!modelAabb.valid()) return false;

    auto lock = rayModel.lockChains();

    bool overflow = false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid) continue;

        std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
        if (!gridWindowFromModelAabb(*grid, modelAabb, iu0, iu1, iv0, iv1)) continue;

        const float radius = radii[axis];
        for (std::uint32_t iv = iv0; iv <= iv1; ++iv)
        {
            if (static_cast<int>(iv) % _stride != 0) continue;
            for (std::uint32_t iu = iu0; iu <= iu1; ++iu)
            {
                if (static_cast<int>(iu) % _stride != 0) continue;
                writeCell(rayModel, axis, iu, iv, radius, style, &overflow);
                if (overflow) return false;
            }
        }
    }

    _set.markDirty();
    return true;
}

} // namespace app
