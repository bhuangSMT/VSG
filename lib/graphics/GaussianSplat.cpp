#include "GaussianSplat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "UcamDebug.h"

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
// inNormal.xyz is the surface normal; inNormal.w packs edgeMask bits
// (bit0=+X … bit5=-Z), an 8-bit edge strength at bits 6–13, and cutFace at
// bit 14 (Disk lighting only).
const char* const splatVertexBody = R"(
layout(push_constant) uniform PushConstants
{
    mat4 projection;
    mat4 modelView;
} pc;

layout(location = 0) in vec4 inCenterRadius;
layout(location = 1) in vec2 inCorner;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec4 inNormal;

layout(location = 0) out vec2 corner;
layout(location = 1) out vec4 color;
layout(location = 2) out vec3 normalEye;
layout(location = 3) out float bell;
#ifdef HARD_DISK_AA
layout(location = 4) flat out float edgeStrength;
layout(location = 5) flat out vec2 edgeDirDisc;
layout(location = 6) flat out float isCutFace;
#endif

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
        bell = 2.77;
#ifdef HARD_DISK_AA
        edgeStrength = 0.0;
        edgeDirDisc = vec2(0.0);
        isCutFace = 0.0;
#endif
        return;
    }

    vec4 centerEye = pc.modelView * vec4(inCenterRadius.xyz, 1.0);

#ifdef HARD_DISK_AA
    // Hard disks keep a fixed eye-space radius (no Gaussian bell widening).
    bell = 1.0;
#else
    // Zoom proxy from eye depth only — not packed radius. Stride densify scales
    // radius ∝ stride while z shrinks on zoom-in, so radius/z stayed flat and
    // bell never moved. Soft when far (fit-to-unit eye ~3–4), sharp when close.
    const float bellMin = 2.77;
    const float bellMax = 10.0;
    const float zSoft = 4.0;
    const float zSharp = 1.0;
    float t = 1.0 - clamp((-centerEye.z - zSharp) / (zSoft - zSharp), 0.0, 1.0);
    bell = mix(bellMin, bellMax, t);

    // Sharper falloff shrinks the visible core; grow the eye-space radius so
    // the half-intensity footprint still meets the neighbour (≈ √(bell/bellMin)).
    radius *= mix(1.0, sqrt(bellMax / bellMin), t);
#endif
    centerEye.xy += inCorner * radius;

#ifdef SPLAT_DEPTH_PREPASS
#ifdef HARD_DISK_AA
    // Slight bias so the colour pass AA rim still passes the depth test.
    centerEye.z -= radius * 0.05;
#else
    // The depth-only pass lays down the near surface half a splat further away
    // than it really is. Every splat sampling that same surface then still
    // passes the depth test in the colour pass and can blend, while the far
    // side of the model is still rejected.
    centerEye.z -= radius * 0.5;
#endif
#endif

    gl_Position = pc.projection * centerEye;

    corner = inCorner;
    color = inColor;

    // The model-view transform carries a uniform scale at most, so its upper
    // 3x3 rotates the normal without shearing it.
    mat3 mv = mat3(pc.modelView);
    normalEye = mv * inNormal.xyz;

#ifdef HARD_DISK_AA
    uint packed = uint(inNormal.w + 0.5);
    uint mask = packed & 63u;
    edgeStrength = float((packed >> 6) & 255u) / 255.0;
    isCutFace = float((packed >> 14) & 1u);

    // Disc corner space matches eye XY. Sum the set axes into one direction so
    // a rim with two creases narrows along their diagonal instead of twice.
    vec2 ax = (mv * vec3(1.0, 0.0, 0.0)).xy;
    vec2 ay = (mv * vec3(0.0, 1.0, 0.0)).xy;
    vec2 az = (mv * vec3(0.0, 0.0, 1.0)).xy;
    vec2 dir = vec2(0.0);
    if ((mask & 1u) != 0u) dir += ax;
    if ((mask & 2u) != 0u) dir -= ax;
    if ((mask & 4u) != 0u) dir += ay;
    if ((mask & 8u) != 0u) dir -= ay;
    if ((mask & 16u) != 0u) dir += az;
    if ((mask & 32u) != 0u) dir -= az;
    edgeDirDisc = dir;
#endif
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
layout(location = 3) in float bell;
#ifdef HARD_DISK_AA
layout(location = 4) flat in float edgeStrength;
layout(location = 5) flat in vec2 edgeDirDisc;
layout(location = 6) flat in float isCutFace;
#endif

layout(location = 0) out vec4 outColor;

void main()
{
#ifdef HARD_DISK_AA
    // Cheap AABB reject, then the full circular footprint. Nothing is clipped
    // or narrowed near a discontinuity: the radius carries a 1.75x margin over
    // half-cellDiag precisely because endpoints thin out at rims and grazing
    // views, so every disc rasterizes and the depth test picks the winner.
    if (max(abs(corner.x), abs(corner.y)) > 1.0) discard;

    float r = length(corner);
#ifdef SPLAT_DEBUG_SHRINK
    // Opt-in A/B: narrow the crease-facing side to sharpen convex corners.
    // Costs coverage along the crease it sharpens, which reopens rim slivers.
    if (edgeStrength > 0.0 && dot(edgeDirDisc, edgeDirDisc) > 1e-12)
    {
        vec2 e = normalize(edgeDirDisc);
        float along = dot(corner, e);                 // positive = toward crease
        float across = dot(corner, vec2(-e.y, e.x));  // along crease, full width
        float a = (along > 0.0) ? along / mix(1.0, 0.65, edgeStrength) : along;
        r = sqrt(a * a + across * across);
    }
#endif
    if (r > 1.0) discard;

#ifdef SPLAT_DEBUG_EDGE
    // Red = how sharp this endpoint thinks it is, green = narrowing is active.
    outColor = vec4(edgeStrength, dot(edgeDirDisc, edgeDirDisc) > 1e-12 ? 1.0 : 0.0, 0.0, 1.0);
    return;
#endif
#ifdef SPLAT_DEBUG_DEPTH
    // Banded ramp: a mark on a different band sits on a farther surface.
    outColor = vec4(fract(gl_FragCoord.z * 512.0), 0.0, 0.0, 1.0);
    return;
#endif
#else
    float r = length(corner);
    if (r > 1.0) discard;
#endif

#ifdef SPLAT_DEPTH_PREPASS
#ifdef HARD_DISK_AA
    // Opaque disk core writes depth; leave the AA rim out of the depth buffer.
    if (r > 0.95) discard;
#else
    // Only the splat's core takes part in the depth pass. The core still
    // reaches the corners of its grid cell, so the near surface stays sealed,
    // but the soft rim is kept out of the depth buffer instead of stamping a
    // hard disc over the splats beside it.
    if (dot(corner, corner) > 0.3) discard;
#endif

    outColor = vec4(0.0);
#else
#ifdef HARD_DISK_AA
    float falloff = 1.0 - smoothstep(0.95, 1.0, r);
#else
    // Windowed Gaussian: subtracting the falloff's value at the quad's edge and
    // rescaling takes it to exactly zero there, so the splat fades out rather
    // than ending on a visible rim. bell comes from the vertex stage (2.77 far
    // → 10 close, from eye depth) so zoomed-in disks read sharper while overlaps
    // still blend.
    float radiusSquared = r * r;
    float edge = exp(-bell);
    float falloff = max(exp(-bell * radiusSquared) - edge, 0.0) / (1.0 - edge);
#endif

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

    float nDotL = dot(normal, lightDir);
#ifdef HARD_DISK_AA
    // Soft wrap only on cut-tagged discs. Stock keeps hard Lambert so the
    // original cylinder still reads as tight metal.
    float diffuse = (isCutFace > 0.5)
        ? clamp(nDotL * 0.35 + 0.65, 0.0, 1.0)
        : max(nDotL, 0.0);
#else
    float diffuse = max(nDotL, 0.0);
#endif
    vec3 halfway = normalize(lightDir + viewDir);

    // Polished steel: a dark body with a tight highlight hot enough to clip,
    // tinted by the albedo rather than white, since a metal reflects its own
    // colour. The ambient term is deliberately tiny — the render target applies
    // an sRGB transfer curve on write, which lifts midtones hard, so anything
    // with a comfortable-looking ambient here comes out as pale fog on screen.
    //
    // No rim term either — a splat's edge is not a silhouette, so lighting it
    // just outlines every splat and frosts the whole surface.
#ifdef HARD_DISK_AA
    float specular;
    vec3 lit;
    if (isCutFace > 0.5)
    {
        // Cut faces: flatter wrap, higher ambient, almost-diffuse spec.
        specular = pow(max(dot(normal, halfway), 0.0), 4.0);
        lit = color.rgb * (0.32 + 0.50 * diffuse)
            + color.rgb * (0.08 * specular);
    }
    else
    {
        specular = pow(max(dot(normal, halfway), 0.0), 24.0);
        lit = color.rgb * (0.12 + 0.70 * diffuse)
            + color.rgb * (0.45 * specular);
    }
#else
    float specular = pow(max(dot(normal, halfway), 0.0), 70.0);
    vec3 lit = color.rgb * (0.12 + 0.55 * diffuse)
             + color.rgb * (1.40 * specular);
#endif

#ifdef HARD_DISK_AA
    // Opaque replace blend: coverage via discard, not alpha.
    if (falloff < 0.01) discard;
#ifdef SPLAT_DEBUG_FLAT
    // Albedo only: if the dark marks survive this, they are coverage or depth,
    // not shading.
    outColor = vec4(color.rgb, 1.0);
#elif defined(SPLAT_DEBUG_NORMAL)
    outColor = vec4(normal * 0.5 + 0.5, 1.0);
#elif defined(SPLAT_DEBUG_DIFFUSE)
    outColor = vec4(vec3(diffuse), 1.0);
#else
    outColor = vec4(lit, 1.0);
#endif
#else
    float alpha = color.a * falloff;
    if (alpha < 0.01) discard;

    // Premultiplied, to match the blend set up on the pipeline.
    outColor = vec4(lit * alpha, alpha);
#endif
#endif
}
)";

const vsg::vec2 cornerOffsets[4] = {{-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f}};

// Both stages branch on SPLAT_DEPTH_PREPASS / HARD_DISK_AA, so stages are built
// from the same body with the defines prepended. UCAM_SPLAT_DEBUG adds one more
// define for the diagnostic outputs; unset leaves the source unchanged.
std::string shaderSource(const char* body, bool depthPrepass, bool hardDisk = false)
{
    std::string source = "#version 450\n#extension GL_ARB_separate_shader_objects : enable\n";
    if (depthPrepass) source += "#define SPLAT_DEPTH_PREPASS 1\n";
    if (hardDisk) source += "#define HARD_DISK_AA 1\n";
    if (const char* debug = ucamSplatDebugDefine())
    {
        source += "#define ";
        source += debug;
        source += " 1\n";
    }
    return source + body;
}

enum class SplatPipelineKind
{
    DepthPrepass,
    ColorBlend,
    OpaqueHardDisk
};

// Splats are drawn twice for Gaussians. The first pass writes only depth, so
// that the far side of the model is hidden; the second blends without touching
// depth. Hard disks are opaque and use a single depth-writing colour pass.
vsg::ref_ptr<vsg::GraphicsPipeline> createPipeline(SplatPipelineKind kind, PointRenderMode mode)
{
    const bool hardDisk = (mode == PointRenderMode::HardDiskWithAA);
    const bool depthPrepass = (kind == SplatPipelineKind::DepthPrepass);
    const bool opaqueHard = (kind == SplatPipelineKind::OpaqueHardDisk);

    auto vertexShader = vsg::ShaderStage::create(
        VK_SHADER_STAGE_VERTEX_BIT, "main",
        shaderSource(splatVertexBody, depthPrepass && !opaqueHard, hardDisk));
    auto fragmentShader = vsg::ShaderStage::create(
        VK_SHADER_STAGE_FRAGMENT_BIT, "main",
        shaderSource(splatFragmentBody, depthPrepass && !opaqueHard, hardDisk));

    // VSG records the projection and model-view matrices into the first 128
    // bytes of push constant space for whichever pipeline is bound.
    vsg::PushConstantRanges pushConstantRanges{{VK_SHADER_STAGE_VERTEX_BIT, 0, 128}};
    auto pipelineLayout = vsg::PipelineLayout::create(vsg::DescriptorSetLayouts{}, pushConstantRanges);

    vsg::VertexInputState::Bindings vertexBindings{
        VkVertexInputBindingDescription{0, 16, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{1, 8, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{2, 16, VK_VERTEX_INPUT_RATE_VERTEX},
        VkVertexInputBindingDescription{3, 16, VK_VERTEX_INPUT_RATE_VERTEX}};

    vsg::VertexInputState::Attributes vertexAttributes{
        VkVertexInputAttributeDescription{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
        VkVertexInputAttributeDescription{1, 1, VK_FORMAT_R32G32_SFLOAT, 0},
        VkVertexInputAttributeDescription{2, 2, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
        VkVertexInputAttributeDescription{3, 3, VK_FORMAT_R32G32B32A32_SFLOAT, 0}};

    // The quads face the camera, but which way round their winding comes out
    // depends on the view, so neither face can be culled.
    auto rasterizationState = vsg::RasterizationState::create();
    rasterizationState->cullMode = VK_CULL_MODE_NONE;

    auto colorBlendState = vsg::ColorBlendState::create();
    if (depthPrepass && !opaqueHard)
    {
        colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {VK_FALSE,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             0}};
    }
    else if (opaqueHard)
    {
        // Replace; hard disks do not soft-blend with neighbours.
        colorBlendState->attachments = vsg::ColorBlendState::ColorBlendAttachments{
            {VK_FALSE,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
             VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                 VK_COLOR_COMPONENT_A_BIT}};
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
    depthStencilState->depthWriteEnable =
        (depthPrepass || opaqueHard) ? VK_TRUE : VK_FALSE;

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
    if (_pointRenderMode == PointRenderMode::HardDiskWithAA)
    {
        if (!_colorPipeline)
            _colorPipeline = createPipeline(SplatPipelineKind::OpaqueHardDisk, _pointRenderMode);
        _depthPipeline = nullptr;
        return;
    }
    if (!_depthPipeline)
        _depthPipeline = createPipeline(SplatPipelineKind::DepthPrepass, _pointRenderMode);
    if (!_colorPipeline)
        _colorPipeline = createPipeline(SplatPipelineKind::ColorBlend, _pointRenderMode);
}

void GaussianSplatSet::rebindPipelines()
{
    if (!_root || !_colorPipeline) return;

    // Drop existing pass groups (keep overlay if present).
    vsg::ref_ptr<vsg::Node> overlay = _overlay;
    _root->children.clear();

    if (_pointRenderMode == PointRenderMode::HardDiskWithAA)
    {
        auto colorGroup = vsg::StateGroup::create();
        colorGroup->add(vsg::BindGraphicsPipeline::create(_colorPipeline));
        if (_draw) colorGroup->addChild(_draw);
        _root->addChild(colorGroup);
    }
    else
    {
        if (!_depthPipeline) ensurePipelines();
        auto depthGroup = vsg::StateGroup::create();
        depthGroup->add(vsg::BindGraphicsPipeline::create(_depthPipeline));
        if (_draw) depthGroup->addChild(_draw);
        _root->addChild(depthGroup);

        auto colorGroup = vsg::StateGroup::create();
        colorGroup->add(vsg::BindGraphicsPipeline::create(_colorPipeline));
        if (_draw) colorGroup->addChild(_draw);
        _root->addChild(colorGroup);
    }
    if (overlay) _root->addChild(overlay);
    _needsCompile = true;
}

void GaussianSplatSet::setPointRenderMode(PointRenderMode mode)
{
    if (mode == _pointRenderMode) return;
    _pointRenderMode = mode;
    _depthPipeline = nullptr;
    _colorPipeline = nullptr;
    if (!_root) return;
    ensurePipelines();
    rebindPipelines();
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
        (*_normals)[i] = vsg::vec4(0.0f, 0.0f, 1.0f, 0.0f);
    }
}

void GaussianSplatSet::replaceLiveDraw(vsg::ref_ptr<vsg::VertexIndexDraw> next)
{
    if (!_root) return;
    for (auto& child : _root->children)
    {
        if (child == _overlay) continue;
        auto* sg = dynamic_cast<vsg::StateGroup*>(child.get());
        if (!sg) continue;
        auto& kids = sg->children;
        const bool had =
            _draw && std::find(kids.begin(), kids.end(), _draw) != kids.end();
        if (!had) continue;
        kids.erase(std::remove(kids.begin(), kids.end(), _draw), kids.end());
        if (next) sg->addChild(next);
    }
    _draw = next;
}

void GaussianSplatSet::bindDrawArrays()
{
    // assignArrays/assignIndices allocate fresh BufferInfos with no vk buffers.
    // Reusing a VertexIndexDraw that CompileManager already visited leaves
    // record() calling indices->buffer->vk() on a null buffer (SIGSEGV at ~0x30).
    auto newDraw = vsg::VertexIndexDraw::create();
    newDraw->assignArrays(vsg::DataList{_centerRadius, _corners, _colors, _normals});
    newDraw->assignIndices(_indices);
    const auto n = (_drawCount < _capacity) ? _drawCount : _capacity;
    newDraw->indexCount = static_cast<std::uint32_t>(n * 6);
    newDraw->instanceCount = 1;

    if (!_root)
    {
        _draw = newDraw;
        ensurePipelines();
        _root = vsg::Group::create();
        rebindPipelines();
        attachOverlay();
        return;
    }

    // Live graph: keep the compiled draw on screen until compile succeeds.
    if (_draw)
    {
        _pendingDraw = newDraw;
        _retiredDraw = nullptr;
        _needsCompile = true;
        applyDrawCount();
        return;
    }

    _draw = newDraw;
    for (auto& child : _root->children)
    {
        auto* sg = dynamic_cast<vsg::StateGroup*>(child.get());
        if (sg) sg->addChild(_draw);
    }
    _needsCompile = true;
    attachOverlay();
}

void GaussianSplatSet::prepareForCompile()
{
    if (!_pendingDraw) return;
    _retiredDraw = _draw;
    replaceLiveDraw(_pendingDraw);
    _pendingDraw = nullptr;
    applyDrawCount();
}

void GaussianSplatSet::revertFailedCompile()
{
    if (!_retiredDraw) return;
    _pendingDraw = _draw;
    replaceLiveDraw(_retiredDraw);
    _retiredDraw = nullptr;
    _needsCompile = true;
    applyDrawCount();
}

void GaussianSplatSet::noteCompiled()
{
    _needsCompile = false;
    _pendingDraw = nullptr;
    _retiredDraw = nullptr;
    _compiledCapacity = _capacity;
}

void GaussianSplatSet::ensureCapacity(std::size_t needed)
{
    if (needed == 0) return;
    if (needed <= _capacity && _root) return;

    ensurePipelines();

    const std::size_t oldCap = _capacity;
    const std::size_t newCap = needed;

    const auto vertexCount = newCap * 4;
    const auto indexCount = newCap * 6;

    auto centerRadius = vsg::vec4Array::create(vertexCount);
    auto corners = vsg::vec2Array::create(vertexCount);
    auto colors = vsg::vec4Array::create(vertexCount);
    auto normals = vsg::vec4Array::create(vertexCount);
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
    _dirtySpans.clear();
    _needsCompile = true;
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
        _needsCompile = false;
        _dirtySpans.clear();
        _centerRadius = nullptr;
        _corners = nullptr;
        _colors = nullptr;
        _normals = nullptr;
        _indices = nullptr;
        _draw = nullptr;
        _pendingDraw = nullptr;
        _retiredDraw = nullptr;
        _compiledCapacity = 0;
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
        _normals = vsg::vec4Array::create(vertexCount);
        _indices = vsg::uintArray::create(indexCount);

        _centerRadius->properties.dataVariance = vsg::DYNAMIC_DATA;
        _colors->properties.dataVariance = vsg::DYNAMIC_DATA;
        _normals->properties.dataVariance = vsg::DYNAMIC_DATA;

        _capacity = splatCount;
        _drawCount = splatCount;
        _dirtySpans.clear();
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
    // normal.w: 6 direction bits, 8-bit strength at <<6, cutFace at bit 14.
    // Peak 63 + 255*64 + 16384 = 32767, exact in float32.
    const auto strengthQ = static_cast<std::uint32_t>(
        std::lround(std::clamp(splat.edgeStrength, 0.0f, 1.0f) * 255.0f));
    const std::uint32_t packed =
        splat.edgeMask | (strengthQ << 6) | (splat.cutFace ? (1u << 14) : 0u);
    const vsg::vec4 packedNormal(splat.normal.x, splat.normal.y, splat.normal.z,
                                 static_cast<float>(packed));
    for (std::size_t k = 0; k < 4; ++k)
    {
        (*_centerRadius)[base + k] =
            vsg::vec4(splat.position.x, splat.position.y, splat.position.z, splat.radius);
        (*_colors)[base + k] = splat.color;
        (*_normals)[base + k] = packedNormal;
    }
}

void GaussianSplatSet::clearSlot(std::size_t index)
{
    Splat empty{};
    empty.radius = 0.0f;
    empty.normal = vsg::vec3(0.0f, 0.0f, 1.0f);
    empty.edgeMask = 0;
    empty.edgeStrength = 0.0f;
    set(index, empty);
}

void GaussianSplatSet::noteDirtySlots(std::uint32_t first, std::uint32_t count)
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

void GaussianSplatSet::markDirty()
{
    _dirtySpans.clear();
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
    std::size_t n = (_drawCount < _capacity) ? _drawCount : _capacity;
    // Pending grow: the live draw still covers only the last compiled prefix.
    if (_pendingDraw && _compiledCapacity > 0 && n > _compiledCapacity)
        n = _compiledCapacity;
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
    // Same as GaussianSplatSet: rebinding BufferInfos on a compiled draw leaves
    // null vk buffers if CompileManager skips the node.
    if (_draw && _root)
    {
        auto& kids = _root->children;
        kids.erase(std::remove(kids.begin(), kids.end(), _draw), kids.end());
        _draw = nullptr;
    }

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
    else
    {
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
        const auto flags = memory->getMemoryPropertyFlags();
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) return false;
        if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) return false;

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

bool GaussianSplatSet::copyDirtySpan(const DirtySpan& span)
{
    if (!_draw || _draw->arrays.size() < 4) return false;
    const auto firstVert = span.first * 4u;
    const auto vertCount = span.count * 4u;
    // Corners are static; only the dynamic per-splat attributes move.
    return copyArrayRange(*_draw->arrays[0], firstVert, vertCount) &&
           copyArrayRange(*_draw->arrays[2], firstVert, vertCount) &&
           copyArrayRange(*_draw->arrays[3], firstVert, vertCount);
}

void GaussianSplatSet::flushDirty()
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
