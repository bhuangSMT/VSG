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

// Why an incremental patch could not be completed. Anything but Ok means the
// caller has to fall back to a full rebuild, so the reasons are worth telling
// apart: they have different fixes.
enum class PatchResult
{
    Ok,
    LayoutChanged, // stride / resolution / grid extents no longer match
    CellTooDense,  // one cell needs more endpoints than maxEndpointsPerCell
    OutOfSpace     // free list cannot satisfy a grow
};

const char* toString(PatchResult result);

class GaussianSplatCache
{
public:
    // Soft per-cell ceiling (not reserved storage). Larger counts force rebuild.
    static constexpr int maxEndpointsPerCell = 64;

    void clear();   // Zero every GPU slot and drop the layout; keep buffers.
    void release(); // Destroy the VSG splat node and GPU arrays (new model / teardown).
    bool empty() const { return _stride == 0; }

    std::size_t capacity() const { return _capacity; }
    std::size_t liveEndpoints() const { return _live; }

    // Packed refill from the displayed RayModel. Reuses the compiled GPU
    // arrays when they already hold enough slots. Issues only the packed live
    // endpoints. skipCutSplats drops cutBegin/cutEnd discs (Inspection window,
    // or Probe / Subtraction / Union leftovers covered by the cut-face overlay).
    vsg::ref_ptr<vsg::Node> rebuild(const RayModel& rayModel,
                                    int stride,
                                    const std::array<float, 3>& radii,
                                    const SplatStyle& style,
                                    bool skipCutSplats = false);

    // Regenerate only cells overlapping modelAabb. Anything but PatchResult::Ok
    // leaves the region partly updated, so the caller has to rebuild.
    PatchResult updateRegion(const RayModel& rayModel,
                             const BoundingBox& modelAabb,
                             int stride,
                             const std::array<float, 3>& radii,
                             const SplatStyle& style,
                             bool skipCutSplats = false);

    // Inspection preview: draw UV quads in sectionAabb on a separate overlay.
    // Does not overwrite the GPU-resident Subtraction/Union cut-face mesh.
    void updateSectionGrid(const RayModel& rayModel, int stride, const BoundingBox& sectionAabb,
                           const vsg::vec4& color);
    // Hide the Inspection overlay; keep GPU cut faces and show them again.
    void clearSectionGrid();

    // Cut face (Subtraction and Union): remesh the dirty UV window (same halo
    // for delete and stitch) and write only those GPU slots.
    void patchCutFace(const RayModel& rayModel, int stride,
                      const BoundingBox& dirtyModelAabb, const vsg::vec4& color);
    // Pack every cut-tagged end at stride into GPU slots (new model / fallback).
    void rebuildCutFace(const RayModel& rayModel, int stride, const vsg::vec4& color);
    // Show the GPU cut-face mesh when stride/resolution still match;
    // otherwise remesh every cut-tagged end at the new display stride.
    void restoreCutFace(const RayModel& rayModel, int stride, const vsg::vec4& color);
    void showCutFace();
    void clearCutFace();
    bool hasCutFace() const { return _cutFaceLive > 0; }

    vsg::ref_ptr<vsg::Node> node() const { return _set.node(); }
    void markDirty();
    // True when rebuild allocated or grew GPU arrays; the viewer must compile.
    bool gpuNeedsCompile() const
    {
        return _gpuNeedsCompile || _section.needsCompile() || _inspectionSection.needsCompile();
    }
    void noteCompiled();

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
    void addFreeRange(std::uint32_t first, std::uint32_t length);
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
    PatchResult updateCell(const RayModel& rayModel,
                           std::size_t axis,
                           std::uint32_t iu,
                           std::uint32_t iv,
                           float radius,
                           const SplatStyle& style);

    struct OverlayVert
    {
        vsg::vec3 pos{0.0f, 0.0f, 0.0f};
        vsg::vec3 normal{0.0f, 0.0f, 1.0f};
    };
    struct OverlayTri
    {
        OverlayVert v[3]{};
        std::uint8_t axis = 0;
    };

    struct CutFaceLayout
    {
        bool present = false;
        std::uint32_t sampledW = 0;
        std::uint32_t sampledH = 0;
    };

    static constexpr int maxCutFaceTrisPerCell = 64;

    void uploadSectionTris(SectionLineSet& dest, const std::vector<OverlayTri>& tris,
                           const vsg::vec4& color);
    void appendCutFaceTris(const RayModel& rayModel, int stride, const BoundingBox& box,
                           std::vector<OverlayTri>& out, bool clipToAabb, int haloCells) const;
    bool cutFaceLayoutMatches(const RayModel& rayModel, int stride) const;
    void ensureCutFaceLayout(const RayModel& rayModel, int stride);
    CellRef& cutFaceRef(std::size_t axis, std::uint32_t su, std::uint32_t sv);
    void addCutFaceFreeRange(std::uint32_t first, std::uint32_t length);
    bool allocCutFaceBlock(std::uint32_t length, std::uint32_t* outFirst);
    void freeCutFaceCell(CellRef& ref);
    bool writeCutFaceCell(std::size_t axis, std::uint32_t su, std::uint32_t sv,
                          const std::vector<OverlayTri>& tris);
    void presentCutFace();

    GaussianSplatSet _set;
    std::array<AxisLayout, 3> _axes{};
    std::array<std::vector<CellRef>, 3> _cellRefs{};
    std::vector<FreeRange> _freeList;
    Point3d _resolution{0.0, 0.0, 0.0};
    int _stride = 0;
    std::size_t _capacity = 0;
    std::size_t _live = 0;
    std::uint32_t _allocEnd = 0; // one past the last allocated slot
    bool _gpuNeedsCompile = false;
    bool _skipCutSplats = false;
    int _cutFaceStride = 0;
    Point3d _cutFaceResolution{0.0, 0.0, 0.0};
    vsg::vec4 _cutFaceColor{0.95f, 0.35f, 0.10f, 1.0f};
    std::array<CutFaceLayout, 3> _cutFaceLayout{};
    std::array<std::vector<CellRef>, 3> _cutFaceRefs{};
    std::vector<FreeRange> _cutFaceFree;
    std::uint32_t _cutFaceAllocEnd = 0;
    std::size_t _cutFaceLive = 0;
    SectionLineSet _section;
    SectionLineSet _inspectionSection;
};

} // namespace app
