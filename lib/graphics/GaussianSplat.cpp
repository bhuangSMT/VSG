#include "GaussianSplat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace app
{

float splatRadiusForSpan(float cellRadius, double modelLength, double cellDiag, int stride)
{
    if (!(cellRadius > 0.0f) || !(cellDiag > 0.0) || stride < 1) return cellRadius;
    if (!(modelLength > 0.0)) return cellRadius;

    const double fittedLength = modelLength * (static_cast<double>(cellRadius) /
                                               (cellDiag * static_cast<double>(stride)));
    const float cap = static_cast<float>(0.5 * fittedLength);
    if (!(cap > 0.0f)) return cellRadius;
    return std::min(cellRadius, cap);
}

namespace
{

// The quad is built around the splat centre in eye space, so it always faces
// the camera. inCenterRadius.w carries the eye-space half-width.
const char* const splatVertexBody = R"(
layout(push_constant) uniform PushConstants
{
    mat4 projection;
    mat4 modelView;
} pc;

layout(location = 0) in vec4 inCenterRadius;
layout(location = 1) in vec2 inCorner;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec3 inNormal;

layout(location = 0) out vec2 corner;
layout(location = 1) out vec4 color;
layout(location = 2) out vec3 normalEye;

void main()
{
    float radius = inCenterRadius.w;
    // Empty cache slots (padding) stay off-screen.
    if (radius <= 0.0)
    {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        corner = vec2(0.0);
        color = vec4(0.0);
        normalEye = vec3(0.0, 0.0, 1.0);
        return;
    }

    vec4 centerEye = pc.modelView * vec4(inCenterRadius.xyz, 1.0);

    // On-screen size as a fraction of half the viewport height. Cap it so
    // zooming in shrinks world-space radius instead of ballooning into beads.
    // 0.12 filled a tenth of the view per splat and read as balls up close;
    // 0.03 keeps the far sheet and only bites when the camera is near.
    float apparent = radius * abs(pc.projection[1][1]) / max(-centerEye.z, 1e-6);
    const float maxApparent = 0.03;
    if (apparent > maxApparent)
        radius *= maxApparent / apparent;

    centerEye.xy += inCorner * radius;

#ifdef SPLAT_DEPTH_PREPASS
    // The depth-only pass lays down the near surface half a splat further away
    // than it really is. Every splat sampling that same surface then still
    // passes the depth test in the colour pass and can blend, while the far
    // side of the model is still rejected.
    centerEye.z -= radius * 0.5;
#endif

    gl_Position = pc.projection * centerEye;

    corner = inCorner;
    color = inColor;

    // The model-view transform carries a uniform scale at most, so its upper
    // 3x3 rotates the normal without shearing it.
    normalEye = mat3(pc.modelView) * inNormal;
}
)";

// corner runs over [-1, 1] across the quad, so dot(corner, corner) is the
// squared radius in units of the splat's half-width. The exponent puts half
// intensity at half the radius: callers size the quad at twice the spacing
// they need covered, so neighbouring splats cross at half intensity and sum to
// a continuous sheet, while the tail is down to ~6% by the quad's edge and the
// discard does not show up as a hard rim.
//
// The colour pass shades each splat as a shallow dome so that the surface reads
// as metal rather than as flat fog; the vertex colour supplies the albedo.
const char* const splatFragmentBody = R"(
layout(location = 0) in vec2 corner;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normalEye;

layout(location = 0) out vec4 outColor;

void main()
{
    float radiusSquared = dot(corner, corner);
    if (radiusSquared > 1.0) discard;

#ifdef SPLAT_DEPTH_PREPASS
    // Only the splat's core takes part in the depth pass. The core still
    // reaches the corners of its grid cell, so the near surface stays sealed,
    // but the soft rim is kept out of the depth buffer instead of stamping a
    // hard disc over the splats beside it.
    if (radiusSquared > 0.3) discard;

    outColor = vec4(0.0);
#else
    // Windowed Gaussian: subtracting the falloff's value at the quad's edge and
    // rescaling takes it to exactly zero there, so the splat fades out rather
    // than ending on a visible rim. Half intensity at half radius (bell 2.77).
    const float bell = 2.77;
    float edge = exp(-bell);
    float falloff = max(exp(-bell * radiusSquared) - edge, 0.0) / (1.0 - edge);

    // Eye space: the camera looks down -z, so the view direction is +z and the
    // key light is fixed relative to the viewer.
    const vec3 viewDir = vec3(0.0, 0.0, 1.0);
    const vec3 lightDir = normalize(vec3(-0.35, 0.45, 0.82));

    // Shading comes entirely from the surface normal the cast recorded, so each
    // splat is flat-shaded and has no internal gradient. That is what keeps the
    // splats from reading as little spheres: a radial gradient in the normal
    // becomes a radial gradient in the highlight, which is exactly what a lit
    // ball looks like. Perturbing the normal with a dome would put that back.
    //
    // Two-sided: boolean cut faces and coarse sweep normals can point either
    // way; flip to face the camera so the surface does not go black.
    vec3 normal = normalEye;
    float nLen = length(normal);
    if (nLen < 1e-6) normal = viewDir;
    else normal /= nLen;
    if (dot(normal, viewDir) < 0.0) normal = -normal;

    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 halfway = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfway), 0.0), 70.0);

    // Polished steel: a dark body with a tight highlight hot enough to clip,
    // tinted by the albedo rather than white, since a metal reflects its own
    // colour. The ambient term is deliberately tiny — the render target applies
    // an sRGB transfer curve on write, which lifts midtones hard, so anything
    // with a comfortable-looking ambient here comes out as pale fog on screen.
    //
    // No rim term either — a splat's edge is not a silhouette, so lighting it
    // just outlines every splat and frosts the whole surface.
    vec3 lit = color.rgb * (0.12 + 0.55 * diffuse)
             + color.rgb * (1.40 * specular);

    float alpha = color.a * falloff;

    // Premultiplied, to match the blend set up on the pipeline.
    outColor = vec4(lit * alpha, alpha);
#endif
}
)";

const vsg::vec2 cornerOffsets[4] = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}};

// Both stages branch on SPLAT_DEPTH_PREPASS, so both are built from the same
// body with the define prepended.
std::string shaderSource(const char* body, bool depthPrepass)
{
    std::string source = "#version 450\n#extension GL_ARB_separate_shader_objects : enable\n";
    if (depthPrepass) source += "#define SPLAT_DEPTH_PREPASS 1\n";
    return source + body;
}

// Splats are drawn twice. The first pass writes only depth, so that the far
// side of the model is hidden; the second blends the Gaussians without touching
// depth, so that neighbouring splats on the near surface all contribute instead
// of the closest one masking the rest. Drawing them in one pass instead leaves
// each splat's soft rim blocking its neighbours, which shows up as dark
// scalloped edges rather than a continuous sheet.
vsg::ref_ptr<vsg::GraphicsPipeline> createPipeline(bool depthPrepass)
{
    auto vertexShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main",
                                                 shaderSource(splatVertexBody, depthPrepass));
    auto fragmentShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                                                   shaderSource(splatFragmentBody, depthPrepass));

    // VSG records the projection and model-view matrices into the first 128
    // bytes of push constant space for whichever pipeline is bound.
    vsg::PushConstantRanges pushConstantRanges{{VK_SHADER_STAGE_VERTEX_BIT, 0, 128}};
    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{}, pushConstantRanges);

    vsg::VertexInputState::Bindings vertexBindings{
        VkVertexInputBindingDescription{0, 16, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{1, 8, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{2, 16, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{3, 12, VK_VERTEX_INPUT_RATE_VERTEX}};

    vsg::VertexInputState::Attributes vertexAttributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
        VkVertexInputAttributeDescription{1, 1, VK_FORMAT_R32G32_SFLOAT, 0},
        VkVertexInputAttributeDescription{2, 2, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
        VkVertexInputAttributeDescription{3, 3, VK_FORMAT_R32G32B32_SFLOAT, 0}};

    // The quads face the camera, but which way round their winding comes out
    // depends on the view, so neither face can be culled.
    auto rasterizationState = vsg::RasterizationState::create();
    rasterizationState->cullMode = VK_CULL_MODE_NONE;

    auto colorBlendState = vsg::ColorBlendState::create();
    if (depthPrepass)
    {
        colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {VK_FALSE,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             0}};
    }
    else
    {
        // Source-over compositing of premultiplied colour. Overlapping splats
        // converge on the surface colour, which is what makes a dense enough
        // set of them read as a continuous sheet rather than separate blobs.
        colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {VK_TRUE,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
             VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                 VK_COLOR_COMPONENT_A_BIT}};
    }

    auto depthStencilState = vsg::DepthStencilState::create();
    depthStencilState->depthTestEnable = VK_TRUE;
    depthStencilState->depthWriteEnable = depthPrepass ? VK_TRUE : VK_FALSE;

    // The multisample state is deliberately left out so that the viewer's own
    // sample count is picked up from the compile context.
    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindings, vertexAttributes),
        vsg::InputAssemblyState::create(),
        rasterizationState,
        colorBlendState,
        depthStencilState};

    return vsg::GraphicsPipeline::create(pipelineLayout,
                                         vsg::ShaderStages{vertexShader, fragmentShader},
                                         pipelineStates);
}

} // namespace

void GaussianSplatSet::ensurePipelines()
{
    if (!_depthPipeline) _depthPipeline = createPipeline(true);
    if (!_colorPipeline) _colorPipeline = createPipeline(false);
}

void GaussianSplatSet::initSlotGeometry(std::size_t beginSplat, std::size_t endSplat)
{
    if (!_corners || !_indices) return;
    for (std::size_t s = beginSplat; s < endSplat; ++s)
    {
        const auto base = s * 4;
        for (std::size_t k = 0; k < 4; ++k)
            (*_corners)[base + k] = cornerOffsets[k];

        const auto first = static_cast<std::uint32_t>(base);
        const auto indexBase = s * 6;
        (*_indices)[indexBase + 0] = first + 0;
        (*_indices)[indexBase + 1] = first + 1;
        (*_indices)[indexBase + 2] = first + 2;
        (*_indices)[indexBase + 3] = first + 0;
        (*_indices)[indexBase + 4] = first + 2;
        (*_indices)[indexBase + 5] = first + 3;
    }
}

void GaussianSplatSet::zeroDynamicRange(std::size_t beginSplat, std::size_t endSplat)
{
    if (!_centerRadius || !_colors || !_normals) return;
    const auto beginVert = beginSplat * 4;
    const auto endVert = endSplat * 4;
    for (std::size_t i = beginVert; i < endVert; ++i)
    {
        (*_centerRadius)[i] = vsg::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        (*_colors)[i] = vsg::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        (*_normals)[i] = vsg::vec3(0.0f, 0.0f, 1.0f);
    }
}

void GaussianSplatSet::bindDrawArrays()
{
    if (!_draw)
        _draw = vsg::VertexIndexDraw::create();

    _draw->assignArrays(vsg::DataList{_centerRadius, _corners, _colors, _normals});
    _draw->assignIndices(_indices);
    applyDrawCount();
    _draw->instanceCount = 1;

    if (!_root)
    {
        ensurePipelines();
        _root = vsg::Group::create();

        auto depthGroup = vsg::StateGroup::create();
        depthGroup->add(vsg::BindGraphicsPipeline::create(_depthPipeline));
        depthGroup->addChild(_draw);
        _root->addChild(depthGroup);

        auto colorGroup = vsg::StateGroup::create();
        colorGroup->add(vsg::BindGraphicsPipeline::create(_colorPipeline));
        colorGroup->addChild(_draw);
        _root->addChild(colorGroup);
    }
    attachOverlay();
}

void GaussianSplatSet::ensureCapacity(std::size_t needed)
{
    if (needed == 0) return;
    if (needed <= _capacity && _root) return;

    ensurePipelines();

    const std::size_t oldCap = _capacity;
    std::size_t newCap = needed;
    if (oldCap > 0)
        newCap = std::max(needed, oldCap + oldCap / 2);

    const auto vertexCount = newCap * 4;
    const auto indexCount = newCap * 6;

    auto centerRadius = vsg::vec4Array::create(vertexCount);
    auto corners = vsg::vec2Array::create(vertexCount);
    auto colors = vsg::vec4Array::create(vertexCount);
    auto normals = vsg::vec3Array::create(vertexCount);
    auto indices = vsg::uintArray::create(indexCount);

    centerRadius->properties.dataVariance = vsg::DYNAMIC_DATA;
    colors->properties.dataVariance = vsg::DYNAMIC_DATA;
    normals->properties.dataVariance = vsg::DYNAMIC_DATA;

    // Preserve existing slots for updateRegion grow; init only the new tail.
    if (oldCap > 0 && _centerRadius && _corners && _colors && _normals && _indices)
    {
        const auto oldVerts = oldCap * 4;
        const auto oldIndices = oldCap * 6;
        std::copy_n(_centerRadius->begin(), oldVerts, centerRadius->begin());
        std::copy_n(_corners->begin(), oldVerts, corners->begin());
        std::copy_n(_colors->begin(), oldVerts, colors->begin());
        std::copy_n(_normals->begin(), oldVerts, normals->begin());
        std::copy_n(_indices->begin(), oldIndices, indices->begin());
    }

    _centerRadius = centerRadius;
    _corners = corners;
    _colors = colors;
    _normals = normals;
    _indices = indices;
    _capacity = newCap;
    if (_drawCount > _capacity) _drawCount = _capacity;

    initSlotGeometry(oldCap, newCap);
    zeroDynamicRange(oldCap, newCap);
    bindDrawArrays();
}

void GaussianSplatSet::resize(std::size_t splatCount)
{
    if (splatCount == 0)
    {
        _capacity = 0;
        _drawCount = 0;
        _centerRadius = nullptr;
        _corners = nullptr;
        _colors = nullptr;
        _normals = nullptr;
        _indices = nullptr;
        _draw = nullptr;
        _root = nullptr;
        // Keep cached pipelines for the next ensureCapacity.
        return;
    }

    if (splatCount == _capacity && _root) return;

    // Exact size for one-shot builds: allocate precisely when empty, otherwise grow.
    if (_capacity == 0)
    {
        ensurePipelines();
        _capacity = 0; // ensureCapacity treats oldCap==0 → newCap==needed
        // Temporarily call grow logic with forced exact size:
        const auto vertexCount = splatCount * 4;
        const auto indexCount = splatCount * 6;

        _centerRadius = vsg::vec4Array::create(vertexCount);
        _corners = vsg::vec2Array::create(vertexCount);
        _colors = vsg::vec4Array::create(vertexCount);
        _normals = vsg::vec3Array::create(vertexCount);
        _indices = vsg::uintArray::create(indexCount);

        _centerRadius->properties.dataVariance = vsg::DYNAMIC_DATA;
        _colors->properties.dataVariance = vsg::DYNAMIC_DATA;
        _normals->properties.dataVariance = vsg::DYNAMIC_DATA;

        _capacity = splatCount;
        _drawCount = splatCount;
        initSlotGeometry(0, splatCount);
        zeroDynamicRange(0, splatCount);
        bindDrawArrays();
        return;
    }

    ensureCapacity(splatCount);
}

void GaussianSplatSet::set(std::size_t index, const Splat& splat)
{
    if (!_centerRadius || index >= _capacity) return;

    const auto base = index * 4;
    for (std::size_t k = 0; k < 4; ++k)
    {
        (*_centerRadius)[base + k] =
            vsg::vec4(splat.position.x, splat.position.y, splat.position.z, splat.radius);
        (*_colors)[base + k] = splat.color;
        (*_normals)[base + k] = splat.normal;
    }
}

void GaussianSplatSet::clearSlot(std::size_t index)
{
    Splat empty{};
    empty.radius = 0.0f;
    empty.normal = vsg::vec3(0.0f, 0.0f, 1.0f);
    set(index, empty);
}

void GaussianSplatSet::markDirty()
{
    if (_centerRadius) _centerRadius->dirty();
    if (_colors) _colors->dirty();
    if (_normals) _normals->dirty();
}

void GaussianSplatSet::setDrawCount(std::size_t splatCount)
{
    _drawCount = splatCount;
    if (_drawCount > _capacity) _drawCount = _capacity;
    applyDrawCount();
}

void GaussianSplatSet::applyDrawCount()
{
    if (!_draw) return;
    const auto n = (_drawCount < _capacity) ? _drawCount : _capacity;
    _draw->indexCount = static_cast<std::uint32_t>(n * 6);
}

void GaussianSplatSet::setOverlay(vsg::ref_ptr<vsg::Node> overlay)
{
    if (_root && _overlay && overlay != _overlay)
    {
        auto& kids = _root->children;
        kids.erase(std::remove(kids.begin(), kids.end(), _overlay), kids.end());
    }
    _overlay = overlay;
    attachOverlay();
}

void GaussianSplatSet::attachOverlay()
{
    if (!_root || !_overlay) return;
    for (auto& child : _root->children)
    {
        if (child == _overlay) return;
    }
    _root->addChild(_overlay);
}

namespace
{

const char* const sectionVertex = R"(
layout(push_constant) uniform PushConstants
{
    mat4 projection;
    mat4 modelView;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 0) out vec4 color;
layout(location = 1) out vec3 normalEye;

void main()
{
    gl_Position = pc.projection * pc.modelView * vec4(inPos, 1.0);
    color = inColor;
    normalEye = mat3(pc.modelView) * inNormal;
}
)";

const char* const sectionFragment = R"(
layout(location = 0) in vec4 color;
layout(location = 1) in vec3 normalEye;
layout(location = 0) out vec4 outColor;

void main()
{
    const vec3 viewDir = vec3(0.0, 0.0, 1.0);
    const vec3 lightDir = normalize(vec3(-0.35, 0.45, 0.82));

    vec3 normal = normalEye;
    float nLen = length(normal);
    if (nLen < 1e-6) normal = viewDir;
    else normal /= nLen;
    if (dot(normal, viewDir) < 0.0) normal = -normal;

    float diffuse = max(dot(normal, lightDir), 0.0);
    vec3 halfway = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfway), 0.0), 70.0);

    vec3 lit = color.rgb * (0.12 + 0.55 * diffuse)
             + color.rgb * (1.40 * specular);
    outColor = vec4(lit, color.a);
}
)";

vsg::ref_ptr<vsg::GraphicsPipeline> createSectionLinePipeline()
{
    auto vertexShader = vsg::ShaderStage::create(VK_SHADER_STAGE_VERTEX_BIT, "main",
                                                 shaderSource(sectionVertex, false));
    auto fragmentShader = vsg::ShaderStage::create(VK_SHADER_STAGE_FRAGMENT_BIT, "main",
                                                   shaderSource(sectionFragment, false));

    vsg::PushConstantRanges pushConstantRanges{{VK_SHADER_STAGE_VERTEX_BIT, 0, 128}};
    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{}, pushConstantRanges);

    vsg::VertexInputState::Bindings vertexBindings{
        VkVertexInputBindingDescription{0, 12, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{1, 16, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{2, 12, VK_VERTEX_INPUT_RATE_VERTEX}};
    vsg::VertexInputState::Attributes vertexAttributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        VkVertexInputAttributeDescription{1, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
        VkVertexInputAttributeDescription{2, 2, VK_FORMAT_R32G32B32_SFLOAT, 0}};

    auto inputAssembly = vsg::InputAssemblyState::create();
    inputAssembly->topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    auto rasterizationState = vsg::RasterizationState::create();
    rasterizationState->cullMode = VK_CULL_MODE_NONE;

    auto depthStencilState = vsg::DepthStencilState::create();
    depthStencilState->depthTestEnable = VK_TRUE;
    depthStencilState->depthWriteEnable = VK_TRUE;

    vsg::GraphicsPipelineStates pipelineStates{
        vsg::VertexInputState::create(vertexBindings, vertexAttributes),
        inputAssembly,
        rasterizationState,
        vsg::ColorBlendState::create(),
        depthStencilState};

    return vsg::GraphicsPipeline::create(pipelineLayout,
                                         vsg::ShaderStages{vertexShader, fragmentShader},
                                         pipelineStates);
}

} // namespace

void SectionLineSet::ensurePipeline()
{
    if (!_pipeline) _pipeline = createSectionLinePipeline();
}

void SectionLineSet::applyDrawCount()
{
    if (!_draw) return;
    const auto n = (_drawCount < _capacity) ? _drawCount : _capacity;
    _draw->indexCount = static_cast<std::uint32_t>(n * 3);
}

void SectionLineSet::bindDraw()
{
    if (!_draw)
        _draw = vsg::VertexIndexDraw::create();

    _draw->assignArrays(vsg::DataList{_positions, _colors, _normals});
    _draw->assignIndices(_indices);
    applyDrawCount();
    _draw->instanceCount = 1;

    if (!_root)
    {
        ensurePipeline();
        _root = vsg::StateGroup::create();
        _root->add(vsg::BindGraphicsPipeline::create(_pipeline));
        _root->addChild(_draw);
        _needsCompile = true;
    }
}

void SectionLineSet::ensureCapacity(std::size_t needed)
{
    if (needed == 0) return;
    if (needed <= _capacity && _root) return;

    ensurePipeline();

    const std::size_t oldCap = _capacity;
    std::size_t newCap = needed;
    if (oldCap > 0)
        newCap = std::max(needed, oldCap + oldCap / 2);

    const auto vertexCount = newCap * 3;
    auto positions = vsg::vec3Array::create(vertexCount);
    auto colors = vsg::vec4Array::create(vertexCount);
    auto normals = vsg::vec3Array::create(vertexCount);
    auto indices = vsg::uintArray::create(vertexCount);

    positions->properties.dataVariance = vsg::DYNAMIC_DATA;
    colors->properties.dataVariance = vsg::DYNAMIC_DATA;
    normals->properties.dataVariance = vsg::DYNAMIC_DATA;

    if (oldCap > 0 && _positions && _colors && _normals && _indices)
    {
        const auto oldVerts = oldCap * 3;
        std::copy_n(_positions->begin(), oldVerts, positions->begin());
        std::copy_n(_colors->begin(), oldVerts, colors->begin());
        std::copy_n(_normals->begin(), oldVerts, normals->begin());
        std::copy_n(_indices->begin(), oldVerts, indices->begin());
    }

    for (std::size_t i = oldCap * 3; i < vertexCount; ++i)
    {
        (*positions)[i] = vsg::vec3(0.0f, 0.0f, 0.0f);
        (*colors)[i] = vsg::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        (*normals)[i] = vsg::vec3(0.0f, 0.0f, 1.0f);
        (*indices)[i] = static_cast<std::uint32_t>(i);
    }

    _positions = positions;
    _colors = colors;
    _normals = normals;
    _indices = indices;
    _capacity = newCap;
    _dirtySpans.clear();
    if (_drawCount > _capacity) _drawCount = _capacity;

    const bool hadRoot = _root != nullptr;
    bindDraw();
    if (hadRoot)
        _needsCompile = true;
}

void SectionLineSet::setTriangle(std::size_t index, const vsg::vec3& a, const vsg::vec3& b,
                                 const vsg::vec3& c, const vsg::vec3& na, const vsg::vec3& nb,
                                 const vsg::vec3& nc, const vsg::vec4& color)
{
    if (!_positions || !_normals || index >= _capacity) return;
    const auto base = index * 3;
    (*_positions)[base] = a;
    (*_positions)[base + 1] = b;
    (*_positions)[base + 2] = c;
    (*_colors)[base] = color;
    (*_colors)[base + 1] = color;
    (*_colors)[base + 2] = color;

    vsg::vec3 face = vsg::cross(b - a, c - a);
    const float faceLen2 = face.x * face.x + face.y * face.y + face.z * face.z;
    if (faceLen2 > 1.0e-20f) face = face / std::sqrt(faceLen2);
    else face = vsg::vec3(0.0f, 0.0f, 1.0f);

    auto orFace = [&](const vsg::vec3& n) {
        const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 < 1.0e-12f) return face;
        return n;
    };
    (*_normals)[base] = orFace(na);
    (*_normals)[base + 1] = orFace(nb);
    (*_normals)[base + 2] = orFace(nc);
    noteDirtyTriangles(static_cast<std::uint32_t>(index), 1);
}

void SectionLineSet::clearTriangle(std::size_t index)
{
    if (!_positions || index >= _capacity) return;
    const auto base = index * 3;
    (*_positions)[base] = (*_positions)[base + 1] = (*_positions)[base + 2] =
        vsg::vec3(0.0f, 0.0f, 0.0f);
    (*_colors)[base] = (*_colors)[base + 1] = (*_colors)[base + 2] =
        vsg::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    (*_normals)[base] = (*_normals)[base + 1] = (*_normals)[base + 2] =
        vsg::vec3(0.0f, 0.0f, 1.0f);
    noteDirtyTriangles(static_cast<std::uint32_t>(index), 1);
}

void SectionLineSet::setDrawCount(std::size_t triangleCount)
{
    _drawCount = triangleCount;
    if (_drawCount > _capacity) _drawCount = _capacity;
    applyDrawCount();
}

void SectionLineSet::noteDirtyTriangles(std::uint32_t first, std::uint32_t count)
{
    if (count == 0) return;

    DirtySpan span{first, count};
    auto it = std::lower_bound(
        _dirtySpans.begin(), _dirtySpans.end(), span,
        [](const DirtySpan& a, const DirtySpan& b) { return a.first < b.first; });
    it = _dirtySpans.insert(it, span);

    while (it + 1 != _dirtySpans.end() && it->first + it->count >= (it + 1)->first)
    {
        const auto nextEnd = (it + 1)->first + (it + 1)->count;
        const auto thisEnd = it->first + it->count;
        it->count = (nextEnd > thisEnd ? nextEnd : thisEnd) - it->first;
        _dirtySpans.erase(it + 1);
    }
    if (it != _dirtySpans.begin())
    {
        auto prev = it - 1;
        if (prev->first + prev->count >= it->first)
        {
            const auto thisEnd = it->first + it->count;
            const auto prevEnd = prev->first + prev->count;
            prev->count = (thisEnd > prevEnd ? thisEnd : prevEnd) - prev->first;
            _dirtySpans.erase(it);
        }
    }
}

namespace
{

bool copyArrayRange(vsg::BufferInfo& info, std::uint32_t firstVert, std::uint32_t vertCount)
{
    if (!info.buffer || !info.data || vertCount == 0) return false;
    if (info.buffer->sizeVulkanData() == 0) return false;

    const auto stride = info.data->stride();
    if (stride == 0) return false;

    const auto bytes = vertCount * stride;
    const auto byteOffset = firstVert * stride;
    if (static_cast<VkDeviceSize>(byteOffset) + static_cast<VkDeviceSize>(bytes) > info.range)
        return false;

    const auto* src = static_cast<const std::uint8_t*>(info.data->dataPointer());
    if (!src) return false;
    src += byteOffset;

    for (std::uint32_t deviceID = 0; deviceID < info.buffer->sizeVulkanData(); ++deviceID)
    {
        vsg::DeviceMemory* memory = info.buffer->getDeviceMemory(deviceID);
        if (!memory) return false;
        if ((memory->getMemoryPropertyFlags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
            return false;

        void* dst = nullptr;
        const VkResult result =
            memory->map(info.buffer->getMemoryOffset(deviceID) + info.offset + byteOffset,
                        bytes, 0, &dst);
        if (result != VK_SUCCESS || !dst) return false;
        std::memcpy(dst, src, bytes);
        memory->unmap();
    }
    return true;
}

} // namespace

bool SectionLineSet::copyDirtySpan(const DirtySpan& span)
{
    if (!_draw || _draw->arrays.size() < 3) return false;
    const auto firstVert = span.first * 3u;
    const auto vertCount = span.count * 3u;
    return copyArrayRange(*_draw->arrays[0], firstVert, vertCount) &&
           copyArrayRange(*_draw->arrays[1], firstVert, vertCount) &&
           copyArrayRange(*_draw->arrays[2], firstVert, vertCount);
}

void SectionLineSet::flushDirty()
{
    if (_dirtySpans.empty()) return;
    if (_needsCompile || !_draw)
    {
        markDirty();
        return;
    }

    for (const DirtySpan& span : _dirtySpans)
    {
        if (!copyDirtySpan(span))
        {
            markDirty();
            return;
        }
    }
    _dirtySpans.clear();
}

void SectionLineSet::markDirty()
{
    _dirtySpans.clear();
    if (_positions) _positions->dirty();
    if (_colors) _colors->dirty();
    if (_normals) _normals->dirty();
}

void SectionLineSet::release()
{
    _capacity = 0;
    _drawCount = 0;
    _needsCompile = false;
    _dirtySpans.clear();
    _positions = nullptr;
    _colors = nullptr;
    _normals = nullptr;
    _indices = nullptr;
    _draw = nullptr;
    _root = nullptr;
}

vsg::ref_ptr<vsg::Node> createGaussianSplatNode(const std::vector<Splat>& splats)
{
    if (splats.empty())
    {
        throw std::invalid_argument("Gaussian splatting requires at least one point.");
    }

    for (const Splat& splat : splats)
    {
        if (!(splat.radius > 0.0f))
            throw std::invalid_argument("Gaussian splat radius must be positive.");
    }

    GaussianSplatSet set;
    set.resize(splats.size());
    for (std::size_t s = 0; s < splats.size(); ++s) set.set(s, splats[s]);
    return set.node();
}

} // namespace app
