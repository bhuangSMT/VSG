// GaussianSplatCache - sparse GPU splat buffers for Ray-GS.
#include "GaussianSplatCache.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <tbb/parallel_for.h>

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

std::uint32_t endpointNeed(const RaySlot& slot)
{
    const auto endpoints = slot.intervalCount * 2u;
    return endpoints;
}

} // namespace

const char* toString(PatchResult result)
{
    switch (result)
    {
    case PatchResult::Ok: return "ok";
    case PatchResult::LayoutChanged: return "layout changed";
    case PatchResult::CellTooDense: return "cell over endpoint ceiling";
    case PatchResult::OutOfSpace: return "free list exhausted";
    }
    return "unknown";
}

void GaussianSplatCache::clear()
{
    // Soft clear: keep GPU capacity/pipelines so the next rebuild stays warm.
    // After updateRegion, used slots are not a packed prefix — _live is only
    // the allocated count — so every slot in the buffer has to be zeroed.
    if (_capacity > 0)
    {
        for (std::size_t i = 0; i < _capacity; ++i)
            _set.clearSlot(i);
        _set.markDirty();
    }

    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _live = 0;
    _allocEnd = 0;
    _gpuNeedsCompile = false;
    _sectionAabb = {};
    _capacity = _set.capacity();
    _set.setDrawCount(0);
}

void GaussianSplatCache::release()
{
    _set.resize(0);
    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _capacity = 0;
    _live = 0;
    _allocEnd = 0;
    _gpuNeedsCompile = true;
    _sectionAabb = {};
}

bool GaussianSplatCache::layoutMatches(const RayModel& rayModel, int stride) const
{
    if (stride != _stride || rayModel.resolution() != _resolution) return false;
    if (_stride == 0 || _capacity == 0 || _set.capacity() == 0) return false;

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
        const auto expected =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        if (_cellRefs[axis].size() != expected) return false;
    }
    return true;
}

GaussianSplatCache::CellRef& GaussianSplatCache::cellRef(std::size_t axis,
                                                         std::uint32_t su,
                                                         std::uint32_t sv)
{
    const AxisLayout& layout = _axes[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cellRefs[axis][flat];
}

const GaussianSplatCache::CellRef& GaussianSplatCache::cellRef(std::size_t axis,
                                                               std::uint32_t su,
                                                               std::uint32_t sv) const
{
    const AxisLayout& layout = _axes[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cellRefs[axis][flat];
}

void GaussianSplatCache::clearSlots(std::uint32_t first, std::uint32_t count)
{
    for (std::uint32_t i = 0; i < count; ++i)
        _set.clearSlot(static_cast<std::size_t>(first + i));
}

void GaussianSplatCache::addFreeRange(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;

    FreeRange range{first, length};
    auto it = std::lower_bound(
        _freeList.begin(), _freeList.end(), range,
        [](const FreeRange& a, const FreeRange& b) { return a.first < b.first; });

    it = _freeList.insert(it, range);

    // Merge with next.
    if (it + 1 != _freeList.end() && it->first + it->length == (it + 1)->first)
    {
        it->length += (it + 1)->length;
        _freeList.erase(it + 1);
    }
    // Merge with previous.
    if (it != _freeList.begin())
    {
        auto prev = it - 1;
        if (prev->first + prev->length == it->first)
        {
            prev->length += it->length;
            _freeList.erase(it);
        }
    }
}

void GaussianSplatCache::freeBlock(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;
    clearSlots(first, length);
    addFreeRange(first, length);

    if (_live >= length) _live -= length;
    else _live = 0;
}

// Free-list only. Growing GPU capacity here would rebind the live VertexIndexDraw
// with uncompiled BufferInfos, so exhaustion has to fall back to a full rebuild.
bool GaussianSplatCache::allocBlock(std::uint32_t length, std::uint32_t* outFirst)
{
    if (length == 0 || !outFirst) return false;

    for (auto it = _freeList.begin(); it != _freeList.end(); ++it)
    {
        if (it->length < length) continue;

        const std::uint32_t first = it->first;
        if (it->length == length)
        {
            _freeList.erase(it);
        }
        else
        {
            it->first += length;
            it->length -= length;
        }
        *outFirst = first;
        _live += length;
        const std::uint32_t end = first + length;
        if (end > _allocEnd) _allocEnd = end;
        return true;
    }
    return false;
}

bool GaussianSplatCache::fillCell(const RayModel& rayModel,
                                  std::size_t axis,
                                  std::uint32_t iu,
                                  std::uint32_t iv,
                                  float radius,
                                  const SplatStyle& style,
                                  CellRef& ref,
                                  bool clearTrailing)
{
    const RayGrid* grid = rayModel.grid(axis);
    if (!grid || ref.first == CellRef::kInvalid || ref.block == 0) return true;

    const RaySlot& slot = grid->at(iu, iv);
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid->sampleU(iu);
    const double v0 = grid->sampleV(iv);
    const double cellDiag = std::sqrt(static_cast<double>(grid->spacingU) *
                                          static_cast<double>(grid->spacingU) +
                                      static_cast<double>(grid->spacingV) *
                                          static_cast<double>(grid->spacingV));
    const int stride = (_stride > 0) ? _stride : 1;

    vsg::vec4 stock = style.stockColor;
    stock.a = style.opacity;
    vsg::vec4 tool = style.toolColor;
    tool.a = style.opacity;

    std::uint32_t written = 0;
    if (!slot.empty())
    {
        auto spans = grid->pool.span(slot);
        for (const Interval& span : spans)
        {
            if (written + 2 > ref.block) break;

            Point3d start{0.0, 0.0, 0.0};
            Point3d end{0.0, 0.0, 0.0};
            start[axis] = grid->fromTick(span.begin);
            end[axis] = grid->fromTick(span.end);
            start[u] = u0;
            end[u] = u0;
            start[v] = v0;
            end[v] = v0;

            const double modelLength = end[axis] - start[axis];
            // 0.5 left gaps (orange cut-face dots sat apart). 0.85 still
            // tightens the hole wall without breaking the sheet.
            constexpr float sectionScale = 0.85f;
            const float startRadius = splatRadiusForSpan(
                _sectionAabb.contains(start) ? radius * sectionScale : radius, modelLength,
                cellDiag, stride);
            const float endRadius = splatRadiusForSpan(
                _sectionAabb.contains(end) ? radius * sectionScale : radius, modelLength,
                cellDiag, stride);

            const vsg::vec4& color = span.fromBoolean() ? tool : stock;
            const auto base = static_cast<std::size_t>(ref.first + written);
            _set.set(base,
                     {vsg::vec3(static_cast<float>(start[0]),
                                static_cast<float>(start[1]),
                                static_cast<float>(start[2])),
                      normalOrAxis(span.beginNormal, axis, true), color, startRadius});
            _set.set(base + 1,
                     {vsg::vec3(static_cast<float>(end[0]),
                                static_cast<float>(end[1]),
                                static_cast<float>(end[2])),
                      normalOrAxis(span.endNormal, axis, false), color, endRadius});
            written += 2;
        }
    }

    ref.count = static_cast<std::uint16_t>(written);
    if (clearTrailing)
    {
        for (std::uint32_t i = written; i < ref.block; ++i)
            _set.clearSlot(static_cast<std::size_t>(ref.first + i));
    }
    return true;
}

PatchResult GaussianSplatCache::updateCell(const RayModel& rayModel,
                                           std::size_t axis,
                                           std::uint32_t iu,
                                           std::uint32_t iv,
                                           float radius,
                                           const SplatStyle& style)
{
    const AxisLayout& layout = _axes[axis];
    if (!layout.present) return PatchResult::Ok;

    const RayGrid* grid = rayModel.grid(axis);
    if (!grid) return PatchResult::Ok;

    const std::uint32_t su = sampledIndex(iu, _stride);
    const std::uint32_t sv = sampledIndex(iv, _stride);
    if (su >= layout.sampledW || sv >= layout.sampledH) return PatchResult::Ok;

    CellRef& ref = cellRef(axis, su, sv);
    const std::uint32_t needed = endpointNeed(grid->at(iu, iv));

    if (needed > static_cast<std::uint32_t>(maxEndpointsPerCell))
        return PatchResult::CellTooDense;

    if (needed == 0)
    {
        if (ref.block != 0 && ref.first != CellRef::kInvalid)
            freeBlock(ref.first, ref.block);
        ref = {};
        return PatchResult::Ok;
    }

    if (needed <= ref.block && ref.first != CellRef::kInvalid)
    {
        // Live accounting: block already counted in _live from alloc/rebuild.
        fillCell(rayModel, axis, iu, iv, radius, style, ref, true);
        return PatchResult::Ok;
    }

    // +2 endpoint slack so a later 1→2 interval split often stays in-block.
    std::uint32_t blockSize = needed + 2;
    if (blockSize > static_cast<std::uint32_t>(maxEndpointsPerCell))
        blockSize = static_cast<std::uint32_t>(maxEndpointsPerCell);
    if (blockSize < needed) return PatchResult::CellTooDense;

    if (ref.block != 0 && ref.first != CellRef::kInvalid)
        freeBlock(ref.first, ref.block);

    std::uint32_t first = 0;
    if (!allocBlock(blockSize, &first))
    {
        // Slack is opportunistic: an exact fit still beats a full rebuild.
        blockSize = needed;
        if (!allocBlock(blockSize, &first))
        {
            ref = {};
            return PatchResult::OutOfSpace;
        }
    }

    ref.first = first;
    ref.block = static_cast<std::uint16_t>(blockSize);
    ref.count = 0;
    if (!fillCell(rayModel, axis, iu, iv, radius, style, ref, true))
    {
        freeBlock(ref.first, ref.block);
        ref = {};
        return PatchResult::OutOfSpace;
    }
    return PatchResult::Ok;
}

vsg::ref_ptr<vsg::Node> GaussianSplatCache::rebuild(const RayModel& rayModel,
                                                    int stride,
                                                    const std::array<float, 3>& radii,
                                                    const SplatStyle& style,
                                                    const BoundingBox& sectionAabb)
{
    _sectionAabb = sectionAabb;
    auto lock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    if (stride < 1) stride = 1;

    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = rayModel.resolution();
    _stride = stride;
    _live = 0;

    std::array<std::vector<std::uint32_t>, 3> needs{};
    std::array<std::vector<std::uint32_t>, 3> prefix{};
    std::size_t live = 0;

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
        layout.packedBase = live;

        const auto cellCount =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        _cellRefs[axis].assign(cellCount, CellRef{});
        needs[axis].assign(cellCount, 0u);

        tbb::parallel_for(std::size_t{0}, cellCount, [&](std::size_t flat)
        {
            const auto su = flat % layout.sampledW;
            const auto sv = flat / layout.sampledW;
            const auto iu = static_cast<std::uint32_t>(su) * static_cast<std::uint32_t>(stride);
            const auto iv = static_cast<std::uint32_t>(sv) * static_cast<std::uint32_t>(stride);
            auto n = endpointNeed(grid->at(iu, iv));
            if (n > static_cast<std::uint32_t>(maxEndpointsPerCell))
                n = static_cast<std::uint32_t>(maxEndpointsPerCell);
            needs[axis][flat] = n;
        });

        prefix[axis].assign(cellCount + 1, 0u);
        for (std::size_t i = 0; i < cellCount; ++i)
            prefix[axis][i + 1] = prefix[axis][i] + needs[axis][i];

        live += prefix[axis][cellCount];
    }

    if (live == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    // Reuse the compiled arrays when they already hold the packed prefix.
    // Growing (or the first alloc) rebinds BufferInfos and needs compile.
    // 2x headroom is only reserved when we have to allocate anyway, so a
    // later patch can grow cells without a new GPU buffer.
    const std::size_t capBefore = _set.capacity();
    const bool hadNode = _set.node() != nullptr;
    if (capBefore < live)
        _set.ensureCapacity(live * 2);
    _gpuNeedsCompile = !hadNode || _set.capacity() != capBefore;
    _capacity = _set.capacity();
    _live = live;
    _allocEnd = 0;
    if (live > 0)
        _allocEnd = static_cast<std::uint32_t>(live);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const float radius = radii[axis];
        const AxisLayout& layout = _axes[axis];
        const auto cellCount =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        const auto axisBase = static_cast<std::uint32_t>(layout.packedBase);

        tbb::parallel_for(std::size_t{0}, cellCount, [&](std::size_t flat)
        {
            const auto n = needs[axis][flat];
            if (n == 0) return;

            const auto su = flat % layout.sampledW;
            const auto sv = flat / layout.sampledW;
            const auto iu = static_cast<std::uint32_t>(su) * static_cast<std::uint32_t>(stride);
            const auto iv = static_cast<std::uint32_t>(sv) * static_cast<std::uint32_t>(stride);

            CellRef& ref = _cellRefs[axis][flat];
            ref.first = axisBase + prefix[axis][flat];
            ref.block = static_cast<std::uint16_t>(n);
            ref.count = 0;
            fillCell(rayModel, axis, iu, iv, radius, style, ref, false);
        });
    }

    // Unused tail is not submitted. Do not walk it: the draw count is `live`.
    if (_capacity > live)
    {
        _freeList.push_back(FreeRange{static_cast<std::uint32_t>(live),
                                      static_cast<std::uint32_t>(_capacity - live)});
    }

    _set.setDrawCount(live);
    _set.markDirty();
    return _set.node();
}

PatchResult GaussianSplatCache::updateRegion(const RayModel& rayModel,
                                             const BoundingBox& modelAabb,
                                             int stride,
                                             const std::array<float, 3>& radii,
                                             const SplatStyle& style,
                                             const BoundingBox& sectionAabb)
{
    if (!layoutMatches(rayModel, stride)) return PatchResult::LayoutChanged;
    if (!modelAabb.valid()) return PatchResult::LayoutChanged;
    _sectionAabb = sectionAabb;

    auto lock = rayModel.lockChains();

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
                const PatchResult cell = updateCell(rayModel, axis, iu, iv, radius, style);
                if (cell != PatchResult::Ok) return cell;
            }
        }
    }

    // Allocated slots can sit past the last packed prefix. Draw through
    // _allocEnd only: the unused tail is not submitted, even if leftover
    // radius is still in memory. Freed holes below _allocEnd were cleared.
    _set.setDrawCount(_allocEnd);
    _set.markDirty();
    return PatchResult::Ok;
}

} // namespace app
