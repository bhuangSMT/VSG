// Draws a point set as view-facing Gaussian splats.
//
// Each point becomes a quad that is turned to face the camera in the vertex
// shader; the fragment shader evaluates an isotropic Gaussian across it and
// fades the splat out towards the edge.
//
// The set is drawn twice: a depth-only pass fixes the near surface, then a
// blended pass composites the Gaussians against it without writing depth. That
// hides the far side of the model while still letting every splat on the near
// surface contribute, which is what makes a dense set read as a continuous
// sheet. Compositing is order dependent, and the order is not corrected by a
// per-frame sort, but it only matters between splats sampling the same surface.
#pragma once

#include <cstddef>
#include <vector>

#include <vsg/all.h>

namespace app
{

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
    vsg::vec3 normal;
    vsg::vec4 color;
    float radius;
};

// Mutable GPU splat buffers used by the incremental cache. Slots may have
// radius <= 0 (padding); the shader clips those away.
class GaussianSplatSet
{
public:
    // Exact size (or clear when 0). Prefer ensureCapacity for rebuild/grow paths.
    void resize(std::size_t splatCount);
    // Grow-only with hysteresis; no-op when needed <= capacity. Keeps pipelines.
    void ensureCapacity(std::size_t needed);

    void set(std::size_t index, const Splat& splat);
    void clearSlot(std::size_t index);
    void markDirty();

    std::size_t capacity() const { return _capacity; }
    vsg::ref_ptr<vsg::Node> node() const { return _root; }
    bool empty() const { return !_root || _capacity == 0; }

private:
    void ensurePipelines();
    void initSlotGeometry(std::size_t beginSplat, std::size_t endSplat);
    void zeroDynamicRange(std::size_t beginSplat, std::size_t endSplat);
    void bindDrawArrays();

    std::size_t _capacity = 0;
    vsg::ref_ptr<vsg::vec4Array> _centerRadius;
    vsg::ref_ptr<vsg::vec2Array> _corners;
    vsg::ref_ptr<vsg::vec4Array> _colors;
    vsg::ref_ptr<vsg::vec3Array> _normals;
    vsg::ref_ptr<vsg::uintArray> _indices;
    vsg::ref_ptr<vsg::VertexIndexDraw> _draw;
    vsg::ref_ptr<vsg::Group> _root;
    vsg::ref_ptr<vsg::GraphicsPipeline> _depthPipeline;
    vsg::ref_ptr<vsg::GraphicsPipeline> _colorPipeline;
};

// Builds the subgraph that draws splats.
//
// Throws std::invalid_argument if there is nothing to draw or any radius is not
// positive.
vsg::ref_ptr<vsg::Node> createGaussianSplatNode(const std::vector<Splat>& splats);

} // namespace app
