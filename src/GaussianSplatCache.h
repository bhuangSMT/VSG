// GaussianSplatCache - sparse GPU splat buffers for Ray-GS.
//
// Live endpoints are packed into a contiguous GaussianSplatSet. Each sampled
// cell stores a CellRef (first, count, block). Freed blocks go onto a coalesced
// range free-list so updateRegion can grow/shrink without a full mesh rebuild.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <vsg/all.h>

#include "BoundingBox.h"
#include "GaussianSplat.h"
#include "RayModel.h"

namespace app
{

struct SplatStyle
{
    vsg::vec4 stockColor{0.48f, 0.50f, 0.54f, 1.0f};
    vsg::vec4 toolColor{0.95f, 0.35f, 0.10f, 1.0f};
    float opacity = 0.5f;
};

class GaussianSplatCache
{
public:
    // Soft per-cell ceiling (not reserved storage). Larger counts force rebuild.
    static constexpr int maxEndpointsPerCell = 64;

    void clear();
    bool empty() const { return _set.empty(); }

    std::size_t capacity() const { return _capacity; }
    std::size_t liveEndpoints() const { return _live; }

    // Full rebuild from the displayed RayModel. Returns the drawable (unfitted).
    // Throws if the model has no intervals.
    vsg::ref_ptr<vsg::Node> rebuild(const RayModel& rayModel,
                                    int stride,
                                    const std::array<float, 3>& radii,
                                    const SplatStyle& style);

    // Regenerate only cells overlapping modelAabb. Returns false when layout
    // no longer matches, free-list cannot satisfy a grow, or a cell exceeds the
    // soft endpoint ceiling (caller should rebuild).
    bool updateRegion(const RayModel& rayModel,
                      const BoundingBox& modelAabb,
                      int stride,
                      const std::array<float, 3>& radii,
                      const SplatStyle& style);

    vsg::ref_ptr<vsg::Node> node() const { return _set.node(); }
    void markDirty() { _set.markDirty(); }

private:
    struct AxisLayout
    {
        bool present = false;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t sampledW = 0;
        std::uint32_t sampledH = 0;
        std::size_t packedBase = 0; // rebuild packing base for this axis
    };

    struct CellRef
    {
        static constexpr std::uint32_t kInvalid = 0xffffffffu;
        std::uint32_t first = kInvalid;
        std::uint16_t count = 0;
        std::uint16_t block = 0;

        bool empty() const { return count == 0 || first == kInvalid; }
    };

    struct FreeRange
    {
        std::uint32_t first = 0;
        std::uint32_t length = 0;
    };

    bool layoutMatches(const RayModel& rayModel, int stride) const;
    CellRef& cellRef(std::size_t axis, std::uint32_t su, std::uint32_t sv);
    const CellRef& cellRef(std::size_t axis, std::uint32_t su, std::uint32_t sv) const;

    void freeBlock(std::uint32_t first, std::uint32_t length);
    bool allocBlock(std::uint32_t length, std::uint32_t* outFirst);
    void clearSlots(std::uint32_t first, std::uint32_t count);

    // Writes live endpoints into an already-allocated CellRef block.
    // Truncates to ref.block if the cell has more intervals than reserved.
    bool fillCell(const RayModel& rayModel,
                  std::size_t axis,
                  std::uint32_t iu,
                  std::uint32_t iv,
                  float radius,
                  const SplatStyle& style,
                  CellRef& ref,
                  bool clearTrailing);

    // updateRegion path: resize CellRef via free-list, then fill.
    bool updateCell(const RayModel& rayModel,
                    std::size_t axis,
                    std::uint32_t iu,
                    std::uint32_t iv,
                    float radius,
                    const SplatStyle& style);

    GaussianSplatSet _set;
    std::array<AxisLayout, 3> _axes{};
    std::array<std::vector<CellRef>, 3> _cellRefs{};
    std::vector<FreeRange> _freeList;
    Point3d _resolution{0.0, 0.0, 0.0};
    int _stride = 0;
    std::size_t _capacity = 0;
    std::size_t _live = 0;
};

} // namespace app
