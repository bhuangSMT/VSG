// GaussianSplatCache - sparse GPU splat buffers for Ray-GS.
#include "GaussianSplatCache.h"

#include <algorithm>
#include <array>
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

void skipSplatEnds(const Interval& span, double modelLength, float cellRadius,
                   bool skipCutSplats, bool& skipStart, bool& skipEnd)
{
    const bool cutStart = span.cutBegin();
    const bool cutEnd = span.cutEnd();
    skipStart = skipCutSplats && cutStart;
    skipEnd = skipCutSplats && cutEnd;
    // Partner stock disc of a leftover shorter than two cell radii sits on
    // the overlay. Hide both ends so the previous GPU slots can be freed.
    if (skipCutSplats && (cutStart || cutEnd) &&
        modelLength <= 2.0 * static_cast<double>(cellRadius))
    {
        skipStart = true;
        skipEnd = true;
    }
}

std::uint32_t endpointNeed(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv,
                           float radius, bool skipCutSplats)
{
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return 0;

    std::uint32_t n = 0;
    for (const Interval& span : grid.pool.span(slot))
    {
        if (!span.hasSolidLength()) continue;
        const double modelLength = grid.fromTick(span.end) - grid.fromTick(span.begin);
        bool skipStart = false;
        bool skipEnd = false;
        skipSplatEnds(span, modelLength, radius, skipCutSplats, skipStart, skipEnd);
        if (!skipStart) ++n;
        if (!skipEnd) ++n;
    }
    return n;
}

struct SectionPoint
{
    double along = 0.0;
    vsg::vec3 pos{0.0f, 0.0f, 0.0f};
    vsg::vec3 normal{0.0f, 0.0f, 1.0f};
};

struct SectionVert
{
    vsg::vec3 pos{0.0f, 0.0f, 0.0f};
    vsg::vec3 normal{0.0f, 0.0f, 1.0f};
};

void collectSectionPoints(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                          std::uint32_t iv, const BoundingBox& box, bool clipToAabb,
                          std::vector<SectionPoint>& out)
{
    out.clear();
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return;

    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid.sampleU(iu);
    const double v0 = grid.sampleV(iv);
    auto spans = grid.pool.span(slot);
    for (const Interval& span : spans)
    {
        if (!span.hasSolidLength()) continue;
        Point3d start{0.0, 0.0, 0.0};
        Point3d end{0.0, 0.0, 0.0};
        start[axis] = grid.fromTick(span.begin);
        end[axis] = grid.fromTick(span.end);
        start[u] = u0;
        end[u] = u0;
        start[v] = v0;
        end[v] = v0;
        // Inspection clips to the cutter AABB so leftover stock ends in that
        // box do not fill it as a square. The cut-face overlay walks a UV
        // window and takes every tagged leftover on those rays (steep faces
        // jump outside the 3D box between adjacent samples).
        if (span.cutBegin() && (!clipToAabb || !box.valid() || box.contains(start)))
        {
            out.push_back({start[axis],
                           vsg::vec3(static_cast<float>(start[0]),
                                     static_cast<float>(start[1]),
                                     static_cast<float>(start[2])),
                           normalOrAxis(span.beginNormal, axis, true)});
        }
        if (span.cutEnd() && (!clipToAabb || !box.valid() || box.contains(end)))
        {
            out.push_back({end[axis],
                           vsg::vec3(static_cast<float>(end[0]),
                                     static_cast<float>(end[1]),
                                     static_cast<float>(end[2])),
                           normalOrAxis(span.endNormal, axis, false)});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const SectionPoint& a, const SectionPoint& b) { return a.along < b.along; });
}

void emitSectionQuad(const std::vector<SectionPoint>& here,
                     const std::vector<SectionPoint>& east,
                     const std::vector<SectionPoint>& north,
                     const std::vector<SectionPoint>& northEast,
                     float maxEdge,
                     std::vector<std::array<SectionVert, 3>>& tris)
{
    if (here.empty() || maxEdge <= 0.0f) return;
    const float maxE2 = maxEdge * maxEdge;

    auto dist2 = [](const vsg::vec3& a, const vsg::vec3& b) {
        const vsg::vec3 d = a - b;
        return d.x * d.x + d.y * d.y + d.z * d.z;
    };
    auto nearest = [&](const std::vector<SectionPoint>& pts,
                       const vsg::vec3& src) -> const SectionPoint* {
        const SectionPoint* best = nullptr;
        float bestD = maxE2;
        for (const SectionPoint& p : pts)
        {
            const float d2 = dist2(p.pos, src);
            if (d2 <= bestD)
            {
                bestD = d2;
                best = &p;
            }
        }
        return best;
    };

    for (const SectionPoint& h : here)
    {
        const SectionPoint* e = nearest(east, h.pos);
        const SectionPoint* n = nearest(north, h.pos);
        if (!e || !n || dist2(e->pos, n->pos) > maxE2) continue;

        const vsg::vec3 predict = e->pos + n->pos - h.pos;
        const SectionPoint* ne = nullptr;
        float bestNe = maxE2;
        for (const SectionPoint& p : northEast)
        {
            if (dist2(p.pos, e->pos) > maxE2 || dist2(p.pos, n->pos) > maxE2) continue;
            const float d2 = dist2(p.pos, predict);
            if (d2 <= bestNe)
            {
                bestNe = d2;
                ne = &p;
            }
        }

        tris.push_back({SectionVert{h.pos, h.normal},
                        SectionVert{e->pos, e->normal},
                        SectionVert{n->pos, n->normal}});
        if (ne)
        {
            tris.push_back({SectionVert{e->pos, e->normal},
                            SectionVert{ne->pos, ne->normal},
                            SectionVert{n->pos, n->normal}});
        }
    }
}

struct SampledWindow
{
    bool valid = false;
    std::uint32_t su0 = 0;
    std::uint32_t su1 = 0;
    std::uint32_t sv0 = 0;
    std::uint32_t sv1 = 0;
};

struct CollectedTri
{
    std::array<SectionVert, 3> verts;
    std::uint8_t axis = 0;
    std::uint32_t su = 0;
    std::uint32_t sv = 0;
};

double vecComponent(const vsg::vec3& p, std::size_t axis)
{
    if (axis == 0) return static_cast<double>(p.x);
    if (axis == 1) return static_cast<double>(p.y);
    return static_cast<double>(p.z);
}

constexpr int kCutFaceHaloCells = 4;
constexpr float kInspectionMaxEdgeScale = 2.5f;
constexpr float kCutFaceMaxEdgeScale = 8.0f;

void growGridWindowByStride(const RayGrid& grid, int stride, int haloCells,
                            std::uint32_t& iu0, std::uint32_t& iu1,
                            std::uint32_t& iv0, std::uint32_t& iv1)
{
    if (haloCells < 1 || stride < 1 || grid.width == 0 || grid.height == 0) return;
    const auto pad = static_cast<std::uint32_t>(stride) * static_cast<std::uint32_t>(haloCells);
    if (iu0 >= pad) iu0 -= pad;
    else iu0 = 0;
    if (iv0 >= pad) iv0 -= pad;
    else iv0 = 0;
    const auto lastU = grid.width - 1;
    const auto lastV = grid.height - 1;
    if (lastU - iu1 >= pad) iu1 += pad;
    else iu1 = lastU;
    if (lastV - iv1 >= pad) iv1 += pad;
    else iv1 = lastV;
}

SampledWindow makeSampledWindow(const RayGrid& grid, const BoundingBox& box, int stride,
                                int haloCells)
{
    SampledWindow w;
    if (stride < 1) return w;

    std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
    if (box.valid())
    {
        if (!gridWindowFromModelAabb(grid, box, iu0, iu1, iv0, iv1)) return w;
        growGridWindowByStride(grid, stride, haloCells, iu0, iu1, iv0, iv1);
    }
    else
    {
        if (grid.width == 0 || grid.height == 0) return w;
        iu1 = grid.width - 1;
        iv1 = grid.height - 1;
    }

    w.valid = true;
    w.su0 = sampledIndex(iu0, stride);
    w.sv0 = sampledIndex(iv0, stride);
    w.su1 = sampledIndex(iu1, stride);
    w.sv1 = sampledIndex(iv1, stride);
    return w;
}

bool sampledWindowContains(const SampledWindow& w, const vsg::vec3& pos, std::size_t axis,
                           const RayGrid& grid, int stride)
{
    if (!w.valid) return false;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const auto su = sampledIndex(grid.indexU(vecComponent(pos, u)), stride);
    const auto sv = sampledIndex(grid.indexV(vecComponent(pos, v)), stride);
    return su >= w.su0 && su <= w.su1 && sv >= w.sv0 && sv <= w.sv1;
}

void collectSectionTris(const RayModel& rayModel, int stride, const BoundingBox& box,
                        bool clipToAabb, int haloCells, std::vector<CollectedTri>& tris)
{
    if (stride < 1) return;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;

        const SampledWindow w = makeSampledWindow(*grid, box, stride, haloCells);
        if (!w.valid) continue;

        const auto sampledW = w.su1 - w.su0 + 1;
        const auto cellCount =
            static_cast<std::size_t>(sampledW) * static_cast<std::size_t>(w.sv1 - w.sv0 + 1);
        std::vector<std::vector<SectionPoint>> cells(cellCount);

        for (std::uint32_t sv = w.sv0; sv <= w.sv1; ++sv)
        {
            const auto iv = sv * static_cast<std::uint32_t>(stride);
            if (iv >= grid->height) continue;
            for (std::uint32_t su = w.su0; su <= w.su1; ++su)
            {
                const auto iu = su * static_cast<std::uint32_t>(stride);
                if (iu >= grid->width) continue;
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                collectSectionPoints(*grid, axis, iu, iv, box, clipToAabb, cells[flat]);
            }
        }

        const double du = grid->spacingU * static_cast<double>(stride);
        const double dv = grid->spacingV * static_cast<double>(stride);
        const float edgeScale = clipToAabb ? kInspectionMaxEdgeScale : kCutFaceMaxEdgeScale;
        const float maxEdge = edgeScale * static_cast<float>(std::sqrt(du * du + dv * dv));

        std::vector<std::array<SectionVert, 3>> quads;
        const auto axisByte = static_cast<std::uint8_t>(axis);
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
            {
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                const auto east = flat + 1;
                const auto north = flat + sampledW;
                const auto northEast = north + 1;
                const auto before = quads.size();
                emitSectionQuad(cells[flat], cells[east], cells[north], cells[northEast],
                                maxEdge, quads);
                for (auto q = before; q < quads.size(); ++q)
                    tris.push_back({quads[q], axisByte, su, sv});
            }
        }
    }
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
    _skipCutSplats = false;
    _capacity = _set.capacity();
    _set.setDrawCount(0);
    clearCutFace();
}

void GaussianSplatCache::release()
{
    _set.resize(0);
    _set.setOverlay(nullptr);
    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _capacity = 0;
    _live = 0;
    _allocEnd = 0;
    _gpuNeedsCompile = true;
    _skipCutSplats = false;
    _cutFaceStride = 0;
    _cutFaceResolution = Point3d{0.0, 0.0, 0.0};
    _cutFaceLayout = {};
    for (auto& refs : _cutFaceRefs) refs.clear();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    _section.release();
    _inspectionSection.release();
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
            if (!span.hasSolidLength()) continue;
            Point3d start{0.0, 0.0, 0.0};
            Point3d end{0.0, 0.0, 0.0};
            start[axis] = grid->fromTick(span.begin);
            end[axis] = grid->fromTick(span.end);
            start[u] = u0;
            end[u] = u0;
            start[v] = v0;
            end[v] = v0;

            const double modelLength = end[axis] - start[axis];
            const float spanRadius = splatRadiusForSpan(radius, modelLength, cellDiag, stride);
            bool skipStart = false;
            bool skipEnd = false;
            skipSplatEnds(span, modelLength, radius, _skipCutSplats, skipStart, skipEnd);

            if (!skipStart)
            {
                if (written + 1 > ref.block) break;
                const auto base = static_cast<std::size_t>(ref.first + written);
                _set.set(base,
                         {vsg::vec3(static_cast<float>(start[0]),
                                    static_cast<float>(start[1]),
                                    static_cast<float>(start[2])),
                          normalOrAxis(span.beginNormal, axis, true),
                          span.cutBegin() ? tool : stock, spanRadius});
                ++written;
            }
            if (!skipEnd)
            {
                if (written + 1 > ref.block) break;
                const auto base = static_cast<std::size_t>(ref.first + written);
                _set.set(base,
                         {vsg::vec3(static_cast<float>(end[0]),
                                    static_cast<float>(end[1]),
                                    static_cast<float>(end[2])),
                          normalOrAxis(span.endNormal, axis, false),
                          span.cutEnd() ? tool : stock, spanRadius});
                ++written;
            }
        }
    }

    ref.count = static_cast<std::uint16_t>(written);
    // Rebuild packs into a reused GPU buffer and still issues the whole
    // block. Skipped cut-face ends (and any unused tail) must be radius 0
    // or leftover dots show through the section mesh.
    if (clearTrailing || _skipCutSplats)
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
    const std::uint32_t needed = endpointNeed(*grid, iu, iv, radius, _skipCutSplats);

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
                                                    bool skipCutSplats)
{
    _skipCutSplats = skipCutSplats;
    auto lock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
    {
        _set.setDrawCount(0);
        _set.markDirty();
        clearSectionGrid();
        return _set.node();
    }

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
            auto n = endpointNeed(*grid, iu, iv, radii[axis], skipCutSplats);
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
    {
        // A boolean can empty every sampled cell while the grids remain.
        // Keep the compiled buffers; issue nothing this frame.
        _set.setDrawCount(0);
        _set.markDirty();
        clearSectionGrid();
        return _set.node();
    }

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

    // Reused arrays still hold the previous packed/patched discs. Zero them
    // before refill so skipped leftovers cannot linger in old GPU slots.
    for (std::size_t i = 0; i < _capacity; ++i)
        _set.clearSlot(i);

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
            fillCell(rayModel, axis, iu, iv, radius, style, ref, true);
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
    presentCutFace();
    return _set.node();
}

PatchResult GaussianSplatCache::updateRegion(const RayModel& rayModel,
                                             const BoundingBox& modelAabb,
                                             int stride,
                                             const std::array<float, 3>& radii,
                                             const SplatStyle& style,
                                             bool skipCutSplats)
{
    if (!layoutMatches(rayModel, stride)) return PatchResult::LayoutChanged;
    if (!modelAabb.valid()) return PatchResult::LayoutChanged;
    _skipCutSplats = skipCutSplats;

    auto lock = rayModel.lockChains();

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid) continue;

        std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
        if (!gridWindowFromModelAabb(*grid, modelAabb, iu0, iu1, iv0, iv1)) continue;
        growGridWindowByStride(*grid, _stride, kCutFaceHaloCells, iu0, iu1, iv0, iv1);

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

void GaussianSplatCache::markDirty()
{
    _set.markDirty();
    _section.markDirty();
    _inspectionSection.markDirty();
}

void GaussianSplatCache::noteCompiled()
{
    _gpuNeedsCompile = false;
    _section.noteCompiled();
    _inspectionSection.noteCompiled();
}

void GaussianSplatCache::clearSectionGrid()
{
    _inspectionSection.setDrawCount(0);
    _inspectionSection.markDirty();
    presentCutFace();
}

void GaussianSplatCache::clearCutFace()
{
    _cutFaceStride = 0;
    _cutFaceResolution = Point3d{0.0, 0.0, 0.0};
    _cutFaceLayout = {};
    for (auto& refs : _cutFaceRefs) refs.clear();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    _section.setDrawCount(0);
    _section.markDirty();
}

void GaussianSplatCache::presentCutFace()
{
    if (_cutFaceLive > 0 && _section.node())
        _set.setOverlay(_section.node());
    else
        _set.setOverlay(nullptr);
}

void GaussianSplatCache::uploadSectionTris(SectionLineSet& dest, const std::vector<OverlayTri>& tris,
                                           const vsg::vec4& color)
{
    if (tris.empty())
    {
        dest.setDrawCount(0);
        dest.markDirty();
        presentCutFace();
        return;
    }

    dest.ensureCapacity(tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i)
        dest.setTriangle(i, tris[i].v[0].pos, tris[i].v[1].pos, tris[i].v[2].pos,
                         tris[i].v[0].normal, tris[i].v[1].normal, tris[i].v[2].normal, color);
    dest.setDrawCount(tris.size());
    dest.markDirty();
    _set.setOverlay(dest.node());
}

void GaussianSplatCache::appendCutFaceTris(const RayModel& rayModel, int stride,
                                           const BoundingBox& box,
                                           std::vector<OverlayTri>& out, bool clipToAabb,
                                           int haloCells) const
{
    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, box, clipToAabb, haloCells, collected);
    out.reserve(out.size() + collected.size());
    for (const auto& t : collected)
    {
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        out.push_back(tri);
    }
}

bool GaussianSplatCache::cutFaceLayoutMatches(const RayModel& rayModel, int stride) const
{
    if (stride < 1 || stride != _cutFaceStride) return false;
    if (rayModel.resolution() != _cutFaceResolution) return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        const CutFaceLayout& layout = _cutFaceLayout[axis];
        if (!grid || grid->empty())
        {
            if (layout.present) return false;
            continue;
        }
        if (!layout.present) return false;
        if (layout.sampledW != sampledCount(grid->width, stride) ||
            layout.sampledH != sampledCount(grid->height, stride))
            return false;
        const auto expected =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        if (_cutFaceRefs[axis].size() != expected) return false;
    }
    return true;
}

void GaussianSplatCache::ensureCutFaceLayout(const RayModel& rayModel, int stride)
{
    _cutFaceStride = stride;
    _cutFaceResolution = rayModel.resolution();
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        CutFaceLayout& layout = _cutFaceLayout[axis];
        if (!grid || grid->empty())
        {
            layout = {};
            _cutFaceRefs[axis].clear();
            continue;
        }
        layout.present = true;
        layout.sampledW = sampledCount(grid->width, stride);
        layout.sampledH = sampledCount(grid->height, stride);
        const auto n =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        _cutFaceRefs[axis].assign(n, CellRef{});
    }
}

GaussianSplatCache::CellRef& GaussianSplatCache::cutFaceRef(std::size_t axis, std::uint32_t su,
                                                            std::uint32_t sv)
{
    const CutFaceLayout& layout = _cutFaceLayout[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cutFaceRefs[axis][flat];
}

void GaussianSplatCache::addCutFaceFreeRange(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;
    FreeRange range{first, length};
    auto it = std::lower_bound(
        _cutFaceFree.begin(), _cutFaceFree.end(), range,
        [](const FreeRange& a, const FreeRange& b) { return a.first < b.first; });
    it = _cutFaceFree.insert(it, range);
    if (it + 1 != _cutFaceFree.end() && it->first + it->length == (it + 1)->first)
    {
        it->length += (it + 1)->length;
        _cutFaceFree.erase(it + 1);
    }
    if (it != _cutFaceFree.begin())
    {
        auto prev = it - 1;
        if (prev->first + prev->length == it->first)
        {
            prev->length += it->length;
            _cutFaceFree.erase(it);
        }
    }
}

bool GaussianSplatCache::allocCutFaceBlock(std::uint32_t length, std::uint32_t* outFirst)
{
    if (length == 0 || !outFirst) return false;

    for (auto it = _cutFaceFree.begin(); it != _cutFaceFree.end(); ++it)
    {
        if (it->length < length) continue;
        const std::uint32_t first = it->first;
        if (it->length == length) _cutFaceFree.erase(it);
        else
        {
            it->first += length;
            it->length -= length;
        }
        *outFirst = first;
        return true;
    }

    _section.ensureCapacity(static_cast<std::size_t>(_cutFaceAllocEnd) + length);
    if (static_cast<std::size_t>(_cutFaceAllocEnd) + length > _section.capacity()) return false;
    *outFirst = _cutFaceAllocEnd;
    _cutFaceAllocEnd += length;
    return true;
}

void GaussianSplatCache::freeCutFaceCell(CellRef& ref)
{
    if (ref.first == CellRef::kInvalid || ref.block == 0)
    {
        ref = {};
        return;
    }
    for (std::uint32_t i = 0; i < ref.block; ++i)
        _section.clearTriangle(static_cast<std::size_t>(ref.first + i));
    addCutFaceFreeRange(ref.first, ref.block);
    if (_cutFaceLive >= ref.count) _cutFaceLive -= ref.count;
    else _cutFaceLive = 0;
    ref = {};
}

bool GaussianSplatCache::writeCutFaceCell(std::size_t axis, std::uint32_t su, std::uint32_t sv,
                                          const std::vector<OverlayTri>& tris)
{
    CellRef& ref = cutFaceRef(axis, su, sv);
    freeCutFaceCell(ref);
    if (tris.empty()) return true;
    if (tris.size() > static_cast<std::size_t>(maxCutFaceTrisPerCell)) return false;

    const auto n = static_cast<std::uint32_t>(tris.size());
    std::uint32_t first = 0;
    if (!allocCutFaceBlock(n, &first)) return false;
    ref.first = first;
    ref.block = static_cast<std::uint16_t>(n);
    ref.count = static_cast<std::uint16_t>(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        const OverlayTri& tri = tris[i];
        _section.setTriangle(static_cast<std::size_t>(first + i), tri.v[0].pos, tri.v[1].pos,
                             tri.v[2].pos, tri.v[0].normal, tri.v[1].normal, tri.v[2].normal,
                             _cutFaceColor);
    }
    _cutFaceLive += n;
    return true;
}

void GaussianSplatCache::updateSectionGrid(const RayModel& rayModel, int stride,
                                           const BoundingBox& sectionAabb,
                                           const vsg::vec4& color)
{
    if (!sectionAabb.valid() || stride < 1)
    {
        clearSectionGrid();
        return;
    }

    auto lock = rayModel.lockChains();
    std::vector<OverlayTri> tris;
    appendCutFaceTris(rayModel, stride, sectionAabb, tris, false, kCutFaceHaloCells);
    uploadSectionTris(_inspectionSection, tris, color);
}

void GaussianSplatCache::rebuildCutFace(const RayModel& rayModel, int stride,
                                        const vsg::vec4& color)
{
    _cutFaceColor = color;
    if (stride < 1)
    {
        clearCutFace();
        presentCutFace();
        return;
    }

    auto lock = rayModel.lockChains();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    ensureCutFaceLayout(rayModel, stride);

    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, BoundingBox{}, false, 0, collected);

    std::array<std::vector<std::vector<OverlayTri>>, 3> buckets;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_cutFaceLayout[axis].present) continue;
        buckets[axis].resize(_cutFaceRefs[axis].size());
    }

    for (const CollectedTri& t : collected)
    {
        if (t.axis > 2 || !_cutFaceLayout[t.axis].present) continue;
        const CutFaceLayout& layout = _cutFaceLayout[t.axis];
        if (t.su >= layout.sampledW || t.sv >= layout.sampledH) continue;
        const auto flat =
            static_cast<std::size_t>(t.sv) * layout.sampledW + static_cast<std::size_t>(t.su);
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        buckets[t.axis][flat].push_back(tri);
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_cutFaceLayout[axis].present) continue;
        const CutFaceLayout& layout = _cutFaceLayout[axis];
        for (std::uint32_t sv = 0; sv < layout.sampledH; ++sv)
        {
            for (std::uint32_t su = 0; su < layout.sampledW; ++su)
            {
                const auto flat =
                    static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
                if (buckets[axis][flat].empty()) continue;
                if (!writeCutFaceCell(axis, su, sv, buckets[axis][flat]))
                {
                    // Rare: a single cell exceeded the soft cap. Keep what we have.
                    break;
                }
            }
        }
    }

    _section.setDrawCount(_cutFaceAllocEnd);
    _section.markDirty();
    presentCutFace();
}

void GaussianSplatCache::patchCutFace(const RayModel& rayModel, int stride,
                                      const BoundingBox& dirtyModelAabb, const vsg::vec4& color)
{
    if (!dirtyModelAabb.valid() || stride < 1) return;
    _cutFaceColor = color;
    if (!cutFaceLayoutMatches(rayModel, stride))
    {
        rebuildCutFace(rayModel, stride, color);
        return;
    }

    auto lock = rayModel.lockChains();
    std::array<SampledWindow, 3> windows{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;
        windows[axis] = makeSampledWindow(*grid, dirtyModelAabb, stride, kCutFaceHaloCells);
        if (!windows[axis].valid) continue;
        const SampledWindow& w = windows[axis];
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
                freeCutFaceCell(cutFaceRef(axis, su, sv));
        }
    }

    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, dirtyModelAabb, false, kCutFaceHaloCells, collected);

    std::array<std::vector<std::vector<OverlayTri>>, 3> buckets;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!windows[axis].valid || !_cutFaceLayout[axis].present) continue;
        const SampledWindow& w = windows[axis];
        const auto sampledW = w.su1 - w.su0;
        const auto sampledH = w.sv1 - w.sv0;
        if (sampledW == 0 || sampledH == 0) continue;
        buckets[axis].resize(static_cast<std::size_t>(sampledW) * static_cast<std::size_t>(sampledH));
    }

    for (const CollectedTri& t : collected)
    {
        if (t.axis > 2 || !windows[t.axis].valid) continue;
        const SampledWindow& w = windows[t.axis];
        if (t.su < w.su0 || t.su >= w.su1 || t.sv < w.sv0 || t.sv >= w.sv1) continue;
        const auto sampledW = w.su1 - w.su0;
        const auto flat = static_cast<std::size_t>(t.sv - w.sv0) * sampledW +
                          static_cast<std::size_t>(t.su - w.su0);
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        buckets[t.axis][flat].push_back(tri);
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!windows[axis].valid || buckets[axis].empty()) continue;
        const SampledWindow& w = windows[axis];
        const auto sampledW = w.su1 - w.su0;
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
            {
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                if (!writeCutFaceCell(axis, su, sv, buckets[axis][flat]))
                {
                    rebuildCutFace(rayModel, stride, color);
                    return;
                }
            }
        }
    }

    _section.setDrawCount(_cutFaceAllocEnd);
    _section.flushDirty();
    presentCutFace();
}

void GaussianSplatCache::showCutFace()
{
    presentCutFace();
}

void GaussianSplatCache::restoreCutFace(const RayModel& rayModel, int stride,
                                        const vsg::vec4& color)
{
    _cutFaceColor = color;
    if (cutFaceLayoutMatches(rayModel, stride))
    {
        showCutFace();
        return;
    }
    rebuildCutFace(rayModel, stride, color);
}

} // namespace app
