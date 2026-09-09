#include "GaussianSplat.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace app
{

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
layout(location = 2) out float closeUp;
layout(location = 3) out vec3 normalEye;

void main()
{
    vec4 centerEye = pc.modelView * vec4(inCenterRadius.xyz, 1.0);

    float radius = inCenterRadius.w;

    // How much of the viewport the splat spans, as a fraction of half its
    // height. This is what "zoomed in" amounts to without having to know the
    // pixel size: projection[1][1] is the vertical scale, and dividing by eye
    // depth gives the on-screen size.
    float apparent = radius * abs(pc.projection[1][1]) / max(-centerEye.z, 1e-6);

    // Once a splat covers a noticeable slice of the screen its own outline
    // starts to resolve and the surface breaks up into beads. Growing it from
    // that point on pushes it further into its neighbours, so the overlap
    // washes the pattern back out into one skin.
    //
    // The bounds are set against what a splat actually spans: a default view of
    // a model sits near 0.09, so the ramp starts just above that and is fully
    // on by a few times closer. Because the measure is the splat's own size, a
    // finer ray resolution rides further up the zoom range before flattening,
    // which is right — smaller splats resolve later.
    closeUp = smoothstep(0.10, 0.32, apparent);
    radius *= 1.0 + 0.90 * closeUp;

    centerEye.xy += inCorner * radius;

#ifdef SPLAT_DEPTH_PREPASS
    // The depth-only pass lays down the near surface half a splat further away
    // than it really is. Every splat sampling that same surface then still
    // passes the depth test in the colour pass and can blend, while the far
    // side of the model is still rejected. Uses the grown radius so the offset
    // keeps pace with the quad.
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
layout(location = 2) in float closeUp;
layout(location = 3) in vec3 normalEye;

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
    // than ending on a visible rim.
    //
    // Zoomed in the bell is broadened, which spreads each splat's weight more
    // evenly over its quad. Together with the extra overlap from the vertex
    // stage that is a wider averaging kernel, so a splat's own profile stops
    // being something the eye can pick out.
    float bell = mix(2.77, 1.35, closeUp);
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

vsg::ref_ptr<vsg::Node> createGaussianSplatNode(const std::vector<Splat>& splats)
{
    if (splats.empty())
    {
        throw std::invalid_argument("Gaussian splatting requires at least one point.");
    }

    const auto splatCount = splats.size();
    const auto vertexCount = splatCount * 4;
    const auto indexCount = splatCount * 6;

    auto centerRadius = vsg::vec4Array::create(vertexCount);
    auto corners = vsg::vec2Array::create(vertexCount);
    auto colors = vsg::vec4Array::create(vertexCount);
    auto normals = vsg::vec3Array::create(vertexCount);
    auto indices = vsg::uintArray::create(indexCount);

    for (std::size_t s = 0; s < splatCount; ++s)
    {
        const Splat& splat = splats[s];
        if (!(splat.radius > 0.0f))
        {
            throw std::invalid_argument("Gaussian splat radius must be positive.");
        }

        const auto base = s * 4;

        for (std::size_t k = 0; k < 4; ++k)
        {
            (*centerRadius)[base + k] =
                vsg::vec4(splat.position.x, splat.position.y, splat.position.z, splat.radius);
            (*corners)[base + k] = cornerOffsets[k];
            (*colors)[base + k] = splat.color;
            (*normals)[base + k] = splat.normal;
        }

        const auto first = static_cast<std::uint32_t>(base);
        const auto indexBase = s * 6;
        (*indices)[indexBase + 0] = first + 0;
        (*indices)[indexBase + 1] = first + 1;
        (*indices)[indexBase + 2] = first + 2;
        (*indices)[indexBase + 3] = first + 0;
        (*indices)[indexBase + 4] = first + 2;
        (*indices)[indexBase + 5] = first + 3;
    }

    // The two passes share one set of buffers and one draw command.
    auto drawCommand = vsg::VertexIndexDraw::create();
    drawCommand->assignArrays(vsg::DataList{centerRadius, corners, colors, normals});
    drawCommand->assignIndices(indices);
    drawCommand->indexCount = static_cast<std::uint32_t>(indices->size());
    drawCommand->instanceCount = 1;

    auto root = vsg::Group::create();
    for (bool depthPrepass : {true, false})
    {
        auto stateGroup = vsg::StateGroup::create();
        stateGroup->add(vsg::BindGraphicsPipeline::create(createPipeline(depthPrepass)));
        stateGroup->addChild(drawCommand);
        root->addChild(stateGroup);
    }

    return root;
}

} // namespace app
