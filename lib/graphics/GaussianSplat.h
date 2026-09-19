// Draws a point set as view-facing Gaussian splats (or opaque hard disks).
//
// Each point becomes a quad that is turned to face the camera in the vertex
// shader; the fragment shader evaluates a Gaussian or a circular hard-disk
// mask across it.
//
// Gaussians are drawn twice: a depth-only pass fixes the near surface, then a
// blended pass composites without writing depth. Hard disks use a single
// opaque depth-writing pass with a full circular footprint, letting the depth
// test pick the winner where discs overlap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vsg/all.h>

namespace app
{

// How each endpoint quad is shaded. Gaussian soft-blends; HardDiskWithAA is a
// nearly opaque disk with a 1–2 px smoothstep rim (interactive surfel look).
enum class PointRenderMode
{
    Gaussian,
    HardDiskWithAA
};

// A single splat. The alpha of colour sets the Gaussian's peak opacity.
//
// radius is the half-width of the splat's quad measured in eye space. The
// corners are offset after the model-view transform has been applied to the
// centre, so radius is in the units that transform maps into rather than in the
// coordinates of position. The Gaussian reaches half intensity at half the
// radius, so the quad extends well past the splat's effective footprint and the
// falloff is not visibly clipped.
// normal is the outward surface normal at the splat, in the same space as
// position. Shading needs it because per-splat shading alone cannot light a
// surface: where splats overlap densely, every pixel ends up covered by some
// splat's highlight, so the highlights average into a flat sheen instead of
// forming one. A normal that follows the surface is what makes the highlight
// coherent and gives the model form.
struct Splat
{
    vsg::vec3 position;
    // Averaged with the neighbours the edgeStrength ramp says are on the same
    // face. Per-bite cuts step the raw normal across the seam where one swept
    // volume meets the next, which shades as a thin dark line.
    vsg::vec3 normal;
    vsg::vec4 color;
    float radius = 0.0f;
    // bit0=+X bit1=-X bit2=+Y bit3=-Y bit4=+Z bit5=-Z normal discontinuity.
    // The set bits sum into one eye-space direction. Narrowing the disc toward
    // it reopens rim slivers, so this now only feeds the debug views.
    std::uint8_t edgeMask = 0;
    // 0 = smooth, 1 = unmistakable crease.
    // Quantised to 8 bits and packed with edgeMask into GPU normal.w.
    float edgeStrength = 0.0f;
};

// cellRadius is already in the space applyFit() maps into (and includes
// stride). modelLength is the interval extent in model space; cellDiag is
// the lateral cell diagonal in the same space. The returned radius never
// exceeds half the fitted span, so a thin beam or a short leftover after a
// cut cannot bloom into empty air.
float splatRadiusForSpan(float cellRadius, double modelLength, double cellDiag, int stride);

// Mutable GPU splat buffers. After a packed rebuild only `setDrawCount`
// slots are issued; padding is not submitted. The shader also clips
// radius <= 0 so a sparse updateRegion path can hide free-list holes.
// updateRegion records dirty slot spans and flushDirty() maps those
// ranges. rebuild / first compile still markDirty() the whole buffer.
class GaussianSplatSet
{
public:
    // Exact size (or clear when 0). Prefer ensureCapacity for rebuild/grow paths.
    void resize(std::size_t splatCount);
    // Grow-only to `needed`; no-op when needed <= capacity. Keeps pipelines.
    void ensureCapacity(std::size_t needed);

    void set(std::size_t index, const Splat& splat);
    void clearSlot(std::size_t index);
    // Slot spans for the sequential patch path. rebuild fills in parallel
    // and must not call this; it markDirty()s the whole buffer instead.
    void noteDirtySlots(std::uint32_t first, std::uint32_t count);
    void flushDirty();
    void markDirty();
    // How many packed slots the draw issues. Unused tail / free-list holes
    // are not submitted, so a leftover radius on the GPU cannot appear.
    void setDrawCount(std::size_t splatCount);

    // Section triangle mesh (Inspection cut face) parented under the same Group.
    void setOverlay(vsg::ref_ptr<vsg::Node> overlay);

    // Swaps fragment falloff (Gaussian vs hard AA disk). Triggers recompile when
    // pipelines already exist.
    void setPointRenderMode(PointRenderMode mode);
    PointRenderMode pointRenderMode() const { return _pointRenderMode; }

    std::size_t capacity() const { return _capacity; }
    vsg::ref_ptr<vsg::Node> node() const { return _root; }
    bool empty() const { return !_root || _capacity == 0; }
    bool needsCompile() const { return _needsCompile || _pendingDraw; }
    void prepareForCompile();
    void revertFailedCompile();
    void noteCompiled();

private:
    struct DirtySpan
    {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    void ensurePipelines();
    void rebindPipelines();
    void initSlotGeometry(std::size_t beginSplat, std::size_t endSplat);
    void zeroDynamicRange(std::size_t beginSplat, std::size_t endSplat);
    void bindDrawArrays();
    void applyDrawCount();
    void replaceLiveDraw(vsg::ref_ptr<vsg::VertexIndexDraw> next);
    void attachOverlay();
    bool copyDirtySpan(const DirtySpan& span);

    std::size_t _capacity = 0;
    std::size_t _drawCount = 0;
    std::size_t _compiledCapacity = 0;
    bool _needsCompile = false;
    PointRenderMode _pointRenderMode = PointRenderMode::Gaussian;
    std::vector<DirtySpan> _dirtySpans;
    vsg::ref_ptr<vsg::vec4Array> _centerRadius;
    vsg::ref_ptr<vsg::vec2Array> _corners;
    vsg::ref_ptr<vsg::vec4Array> _colors;
    // xyz = surface normal, w = edgeMask bits | (edgeStrength << 6) as float.
    vsg::ref_ptr<vsg::vec4Array> _normals;
    vsg::ref_ptr<vsg::uintArray> _indices;
    vsg::ref_ptr<vsg::VertexIndexDraw> _draw;
    vsg::ref_ptr<vsg::VertexIndexDraw> _pendingDraw;
    vsg::ref_ptr<vsg::VertexIndexDraw> _retiredDraw;
    vsg::ref_ptr<vsg::Group> _root;
    vsg::ref_ptr<vsg::GraphicsPipeline> _depthPipeline;
    vsg::ref_ptr<vsg::GraphicsPipeline> _colorPipeline;
    vsg::ref_ptr<vsg::Node> _overlay;
};

// Mutable TRIANGLE_LIST overlay. Inspection replaces the draw in the current
// cutter window; Subtraction/Union patch a GPU-resident cut-face list. The node
// is a child of GaussianSplatSet. Arrays are DYNAMIC_DATA so mouse-move only
// dirty()s after the first compile.
class SectionLineSet
{
public:
    void ensureCapacity(std::size_t triangleCount);
    void setTriangle(std::size_t index, const vsg::vec3& a, const vsg::vec3& b, const vsg::vec3& c,
                     const vsg::vec3& na, const vsg::vec3& nb, const vsg::vec3& nc,
                     const vsg::vec4& color);
    void clearTriangle(std::size_t index);
    void setDrawCount(std::size_t triangleCount);
    void noteDirtyTriangles(std::uint32_t first, std::uint32_t count);
    void flushDirty();
    void markDirty();
    void release();

    std::size_t capacity() const { return _capacity; }
    vsg::ref_ptr<vsg::Node> node() const { return _root; }
    bool needsCompile() const { return _needsCompile; }
    void noteCompiled() { _needsCompile = false; }

private:
    struct DirtySpan
    {
        std::uint32_t first = 0;
        std::uint32_t count = 0;
    };

    void ensurePipeline();
    void bindDraw();
    void applyDrawCount();
    bool copyDirtySpan(const DirtySpan& span);

    std::size_t _capacity = 0;
    std::size_t _drawCount = 0;
    bool _needsCompile = false;
    std::vector<DirtySpan> _dirtySpans;
    vsg::ref_ptr<vsg::vec3Array> _positions;
    vsg::ref_ptr<vsg::vec4Array> _colors;
    vsg::ref_ptr<vsg::vec3Array> _normals;
    vsg::ref_ptr<vsg::uintArray> _indices;
    vsg::ref_ptr<vsg::VertexIndexDraw> _draw;
    vsg::ref_ptr<vsg::StateGroup> _root;
    vsg::ref_ptr<vsg::GraphicsPipeline> _pipeline;
};

// Builds the subgraph that draws splats.
//
// Throws std::invalid_argument if there is nothing to draw or any radius is not
// positive.
vsg::ref_ptr<vsg::Node> createGaussianSplatNode(const std::vector<Splat>& splats);

} // namespace app
