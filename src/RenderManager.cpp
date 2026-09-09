#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "RenderManager.h"

#include "BRep.h"
#include "BooleanOp.h"
#include "ModelPick.h"
#include "Parameter.h"
#include "RayBoolean.h"
#include "SweptVolume.h"
#include "ToolGeometry.h"

namespace app
{

RenderManager::RenderManager(vsg::ref_ptr<vsgQt::Viewer> viewer,
                             vsg::ref_ptr<vsg::Group> scene,
                             vsg::ref_ptr<vsg::Options> options) :
    _viewer(viewer),
    _scene(scene),
    _options(options)
{
    if (!_scene) throw std::runtime_error("RenderManager requires a valid scene root.");
}

RenderManager::~RenderManager()
{
    joinRayCleanup();
}

vsg::ref_ptr<vsg::Node> RenderManager::buildDrawable(vsg::ref_ptr<vsg::vec3Array> positions,
                                                     vsg::ref_ptr<vsg::vec3Array> normals,
                                                     vsg::ref_ptr<vsg::vec4Array> colors,
                                                     VkVertexInputRate colorRate,
                                                     vsg::ref_ptr<vsg::uintArray> indices,
                                                     bool lines,
                                                     bool transparent) const
{
    auto texcoords = vsg::vec2Array::create(positions->size(), vsg::vec2(0.0f, 0.0f));

    // Graphics pipeline via the standard phong shader set.
    auto shaderSet = vsg::createPhongShaderSet(_options);
    auto gpc = vsg::GraphicsPipelineConfigurator::create(shaderSet);

    if (const auto& materialBinding = shaderSet->getDescriptorBinding("material"))
    {
        vsg::ref_ptr<vsg::Data> mat = materialBinding.data;
        if (!mat) mat = vsg::PhongMaterialValue::create();
        gpc->assignDescriptor("material", mat);
    }

    gpc->enableArray("vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, 8);
    gpc->enableArray("vsg_Color", colorRate, 16);

    if (lines || transparent)
    {
        // Drawing lines instead of triangles: switch the topology and stop
        // back-face culling from hiding the far side of the model. Transparent
        // swept volumes also need two-sided rasterization and alpha blending.
        // The shader set's default states are cloned per configurator, so
        // mutating them here does not leak into other pipelines.
        struct SetDrawStates : public vsg::Visitor
        {
            bool lines = false;
            bool transparent = false;

            void apply(vsg::Object& object) override { object.traverse(*this); }
            void apply(vsg::InputAssemblyState& ias) override
            {
                if (lines) ias.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            }
            void apply(vsg::RasterizationState& rs) override
            {
                if (lines || transparent) rs.cullMode = VK_CULL_MODE_NONE;
            }
            void apply(vsg::ColorBlendState& cbs) override
            {
                if (transparent) cbs.configureAttachments(true);
            }
            void apply(vsg::DepthStencilState& dss) override
            {
                if (transparent) dss.depthWriteEnable = VK_FALSE;
            }
        } setDrawStates;
        setDrawStates.lines = lines;
        setDrawStates.transparent = transparent;

        gpc->accept(setDrawStates);
    }

    gpc->init();

    auto vid = vsg::VertexIndexDraw::create();
    vsg::DataList arrays{positions, normals, texcoords, colors};
    vid->assignArrays(arrays);
    vid->assignIndices(indices);
    vid->indexCount = static_cast<uint32_t>(indices->size());
    vid->instanceCount = 1;

    vsg::StateCommands stateCommands;
    if (!gpc->copyTo(stateCommands))
        throw std::runtime_error("Failed to create graphics pipeline.");

    auto stateGroup = vsg::StateGroup::create();
    stateGroup->stateCommands.swap(stateCommands);
    stateGroup->prototypeArrayState = gpc->getSuitableArrayState();
    stateGroup->addChild(vid);

    return stateGroup;
}

vsg::ref_ptr<vsg::Node> RenderManager::applyFit(vsg::ref_ptr<vsg::Node> node,
                                                const BoundingBox& bounds) const
{
    if (!_fitToUnitBox || !bounds.valid()) return node;

    auto transform = vsg::MatrixTransform::create();
    transform->matrix = fitMatrix(bounds);
    transform->addChild(node);

    return transform;
}

vsg::dmat4 RenderManager::fitMatrix(const BoundingBox& bounds) const
{
    if (!_fitToUnitBox || !bounds.valid()) return vsg::dmat4{};

    const Point3d c = bounds.centre();
    const double maxExtent = std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
    const double s = (maxExtent > 0.0) ? 1.0 / maxExtent : 1.0;

    return vsg::scale(s, s, s) * vsg::translate(-c[0], -c[1], -c[2]);
}

vsg::ref_ptr<vsg::Node> RenderManager::createNode(const BRep& brep) const
{
    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();

    const std::size_t vcount = verts.size();
    if (vcount == 0 || brep.faceCount() == 0)
        throw std::runtime_error("Mesh contains no drawable triangles.");

    auto positions = vsg::vec3Array::create(vcount);
    for (std::size_t i = 0; i < vcount; ++i) (*positions)[i] = verts[i];

    // Per-vertex normals via area-weighted face-normal accumulation.
    auto normals = vsg::vec3Array::create(vcount, vsg::vec3(0.0f, 0.0f, 0.0f));
    const std::size_t faces = brep.faceCount();
    for (std::size_t f = 0; f < faces; ++f)
    {
        const std::uint32_t s = faceOffsets[f];
        if (faceOffsets[f + 1] - s < 3) continue;
        const std::uint32_t ia = faceVertices[s];
        const std::uint32_t ib = faceVertices[s + 1];
        const std::uint32_t ic = faceVertices[s + 2];
        const vsg::vec3 fn = vsg::cross(verts[ib] - verts[ia], verts[ic] - verts[ia]);
        (*normals)[ia] += fn;
        (*normals)[ib] += fn;
        (*normals)[ic] += fn;
    }
    for (auto& n : *normals)
    {
        const float len = vsg::length(n);
        n = (len > 0.0f) ? n / len : vsg::vec3(0.0f, 0.0f, 1.0f);
    }

    // Wireframe draws the unique topological edges of the BRep as a line list;
    // Facet draws the triangles themselves.
    const bool wireframe = (_viewMode == ViewMode::Wireframe);

    auto colors = vsg::vec4Array::create(1, wireframe ? _wireframeColor : _surfaceColor);

    const std::vector<std::uint32_t> edgeIndices =
        wireframe ? brep.extractEdgeIndices() : std::vector<std::uint32_t>{};
    const std::vector<std::uint32_t>& sourceIndices = wireframe ? edgeIndices : faceVertices;

    if (sourceIndices.empty())
        throw std::runtime_error("Mesh contains no drawable geometry for the selected view mode.");

    auto indices = vsg::uintArray::create(sourceIndices.size());
    for (std::size_t i = 0; i < sourceIndices.size(); ++i) (*indices)[i] = sourceIndices[i];

    auto drawable = buildDrawable(positions, normals, colors,
                                  VK_VERTEX_INPUT_RATE_INSTANCE, indices, wireframe);

    return applyFit(drawable, BoundingBox::fromBRep(brep));
}

vsg::ref_ptr<vsg::Node> RenderManager::createRayNode(const RayModel& rayModel) const
{
    auto chainLock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    // A fine cast holds far more rays than are worth drawing, so whole grid
    // layers are dropped until what is left fits the budget.
    const int stride = rayModel.strideForRayBudget(maxRenderedRays);
    const std::size_t rays = rayModel.rayCountAtStride(stride);

    // Two endpoints per ray, drawn as an independent line segment.
    const std::size_t pointCount = rays * 2;

    auto positions = vsg::vec3Array::create(pointCount);
    auto normals = vsg::vec3Array::create(pointCount, vsg::vec3(0.0f, 0.0f, 1.0f));
    auto colors = vsg::vec4Array::create(pointCount);
    auto indices = vsg::uintArray::create(pointCount);

    // Prefix-sum write slots per kept chain so workers can fill disjoint ranges.
    struct ChainWrite
    {
        std::size_t axis = 0;
        const RayChain* chain = nullptr;
        std::size_t pointOffset = 0;
    };

    std::vector<ChainWrite> writes;
    writes.reserve(rayModel.chainCount());
    std::size_t next = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (const auto& chain : rayModel.chains(axis))
        {
            if (chain.u % stride != 0 || chain.v % stride != 0) continue;
            if (chain.rays.empty()) continue;

            writes.push_back(ChainWrite{axis, &chain, next});
            next += chain.rays.size() * 2;
        }
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, writes.size()),
        [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i)
            {
                const ChainWrite& write = writes[i];
                const vsg::vec4& axisColor = _rayColors[write.axis];
                std::size_t point = write.pointOffset;

                for (const auto& ray : write.chain->rays)
                {
                    const vsg::vec4& color = ray.fromBoolean ? _toolColor : axisColor;

                    (*positions)[point] = vsg::vec3(static_cast<float>(ray.startPoint[0]),
                                                    static_cast<float>(ray.startPoint[1]),
                                                    static_cast<float>(ray.startPoint[2]));
                    (*colors)[point] = color;
                    ++point;

                    (*positions)[point] = vsg::vec3(static_cast<float>(ray.endPoint[0]),
                                                    static_cast<float>(ray.endPoint[1]),
                                                    static_cast<float>(ray.endPoint[2]));
                    (*colors)[point] = color;
                    ++point;
                }
            }
        });

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, pointCount),
        [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i)
                (*indices)[i] = static_cast<std::uint32_t>(i);
        });

    // Per-vertex colours so each axis keeps its own colour in one draw call.
    auto drawable = buildDrawable(positions, normals, colors,
                                  VK_VERTEX_INPUT_RATE_VERTEX, indices, true);

    return applyFit(drawable, rayModel.bounds());
}

float RenderManager::splatRadius(const RayModel& rayModel, std::size_t axis) const
{
    const Point3d& resolution = rayModel.resolution();

    // Rays running along one axis are cast on a grid over the other two, so it
    // is those two spacings that decide how far apart the endpoints land. Half
    // the grid cell's diagonal is the smallest radius that still reaches the
    // cell's corners, i.e. the point at which the splats stop leaving gaps.
    const double du = resolution[(axis + 1) % 3];
    const double dv = resolution[(axis + 2) % 3];
    double radius = 0.5 * std::sqrt(du * du + dv * dv);

    // The shader puts half intensity at half the quad, so the quad is sized at
    // twice the spacing to be covered: neighbouring splats then meet at half
    // intensity and add up to an unbroken sheet.
    radius *= 2.0;

    // applyFit() scales the model down into a unit box, but a splat's corners
    // are offset in eye space, after that transform. So the radius has to be
    // scaled by hand to match.
    if (_fitToUnitBox && rayModel.bounds().valid())
    {
        const BoundingBox& bounds = rayModel.bounds();
        const double maxExtent = std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
        if (maxExtent > 0.0) radius /= maxExtent;
    }

    return static_cast<float>(radius);
}

vsg::ref_ptr<vsg::Node> RenderManager::createSplatNode(const RayModel& rayModel) const
{
    auto chainLock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    // As in createRayNode: only every stride'th grid line in each direction is
    // drawn, which is what keeps a fine cast inside what the GPU can hold.
    const int stride = rayModel.strideForRayBudget(maxRenderedRays);

    std::vector<Splat> splats;
    splats.reserve(rayModel.rayCountAtStride(stride) * 2);

    // One albedo for every uncut splat. Spans rewritten by boolean use the
    // tool colour so the cut reads against the metal grey.
    vsg::vec4 color = _splatColor;
    color.a = _splatOpacity;

    vsg::vec4 toolColor = _toolColor;
    toolColor.a = _splatOpacity;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        // Dropping layers spaces the endpoints stride times further apart, so
        // the splats have to grow to match or the surface opens up into gaps.
        const float radius = splatRadius(rayModel, axis) * static_cast<float>(stride);

        // The cast recorded the normal of the face met at each end of a span,
        // already pointing out of the solid, so the splats can be shaded by the
        // real surface rather than by the six axis directions.
        for (const auto& chain : rayModel.chains(axis))
        {
            if (chain.u % stride != 0 || chain.v % stride != 0) continue;

            for (const auto& ray : chain.rays)
            {
                auto splatNormal = [](const Normal3f& n) {
                    const float len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                    if (len2 < 1.0e-12f) return vsg::vec3(0.0f, 0.0f, 1.0f);
                    const float inv = 1.0f / std::sqrt(len2);
                    return vsg::vec3(n[0] * inv, n[1] * inv, n[2] * inv);
                };

                const vsg::vec4& splatColor = ray.fromBoolean ? toolColor : color;

                splats.push_back({vsg::vec3(static_cast<float>(ray.startPoint[0]),
                                            static_cast<float>(ray.startPoint[1]),
                                            static_cast<float>(ray.startPoint[2])),
                                  splatNormal(ray.startNormal),
                                  splatColor, radius});
                splats.push_back({vsg::vec3(static_cast<float>(ray.endPoint[0]),
                                            static_cast<float>(ray.endPoint[1]),
                                            static_cast<float>(ray.endPoint[2])),
                                  splatNormal(ray.endNormal),
                                  splatColor, radius});
            }
        }
    }

    auto drawable = createGaussianSplatNode(splats);

    return applyFit(drawable, rayModel.bounds());
}

void RenderManager::showBRep(const BRep& brep)
{
    _current = brep;
    clearRayModels();
    clearSweptVolume();
    rebuild();

    // A new model may change units; the caller reseeds Parameter::toolRadius
    // first, then this rebuilds the mesh if a cutter is active.
    if (_toolType != ToolType::None) rebuildTool(true);
}

void RenderManager::addBRep(const BRep& brep)
{
    attach(createNode(brep), false);
}

void RenderManager::setRayModel(RayModel model)
{
    joinRayCleanup();

    const Point3d resolution = model.resolution();
    const std::size_t rays = model.rayCount();

    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();

    const auto replaced = _rayModels.find(resolution);
    if (replaced != _rayModels.end()) _cachedRays -= replaced->second.rayCount();

    _rayModels.insert_or_assign(resolution, std::move(model));
    _cachedRays += rays;

    touchRayModel(resolution);
    _sourceRayModel = &_rayModels.at(resolution);
    applyBooleanToRayModel();
}

bool RenderManager::useCachedRayModel(const Point3d& resolution)
{
    if (_rayModels.find(resolution) == _rayModels.end()) return false;

    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();
    touchRayModel(resolution);
    _sourceRayModel = &_rayModels.at(resolution);
    applyBooleanToRayModel();
    return true;
}

void RenderManager::touchRayModel(const Point3d& resolution)
{
    const auto previous =
        std::find(_rayModelOrder.begin(), _rayModelOrder.end(), resolution);
    if (previous != _rayModelOrder.end()) _rayModelOrder.erase(previous);
    _rayModelOrder.push_back(resolution);

    // The resolution just touched sits at the back, so it is never the one
    // dropped here, however far over budget a single model puts the cache.
    while (_rayModelOrder.size() > 1 &&
           (_rayModelOrder.size() > maxCachedRayModels || _cachedRays > maxCachedRays))
    {
        const auto oldest = _rayModels.find(_rayModelOrder.front());
        if (oldest != _rayModels.end())
        {
            _cachedRays -= oldest->second.rayCount();
            _rayModels.erase(oldest);
        }
        _rayModelOrder.erase(_rayModelOrder.begin());
    }
}

void RenderManager::clearRayModels()
{
    joinRayCleanup();

    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();
    _rayModels.clear();
    _rayModelOrder.clear();
    _cachedRays = 0;
}

void RenderManager::joinRayCleanup()
{
    if (_rayCleanupThread.joinable()) _rayCleanupThread.join();
}

RayModel* RenderManager::mutableDisplayedRayModel()
{
    if (!_rayModel) return nullptr;
    if (_booleanRayModel && _rayModel == &*_booleanRayModel) return &*_booleanRayModel;

    for (auto& entry : _rayModels)
    {
        if (&entry.second == _rayModel) return &entry.second;
    }
    return nullptr;
}

void RenderManager::setViewMode(ViewMode mode)
{
    if (mode == _viewMode) return;
    _viewMode = mode;
    rebuild();
}

void RenderManager::rebuild()
{
    joinRayCleanup();

    // The ray modes need a model to draw; without one fall back to the surface
    // so the viewport never goes blank.
    if (usesRayModel(_viewMode) && _rayModel && _rayModel->rayCount() > 0)
    {
        if (RayModel* model = mutableDisplayedRayModel())
        {
            _rayCleanupThread = std::thread([model]() { model->removeDegenerateRays(); });
            joinRayCleanup();
        }

        attach(_viewMode == ViewMode::RayGS ? createSplatNode(*_rayModel)
                                            : createRayNode(*_rayModel),
               true);
        return;
    }

    if (_current) attach(createNode(*_current), true);
}

void RenderManager::clear()
{
    _scene->children.clear();
    _modelNode = nullptr;
    _toolTransform = nullptr;
    _sweptNode = nullptr;
    _sweptVolume.reset();
    _current.reset();
    clearRayModels();
    if (_viewer) _viewer->request();
}

void RenderManager::setToolType(ToolType type)
{
    if (type == _toolType) return;
    _toolType = type;
    if (type == ToolType::None)
    {
        clearSweptVolume();
    }
    else
    {
        if (!_sweptVolume) _sweptVolume = SweptVolume{};
        else _sweptVolume->clearLastPose();
    }
    rebuildTool(false);
}

void RenderManager::updateToolGeometry()
{
    rebuildTool(true);
}

float RenderManager::worldToolRadius() const
{
    double radius = Parameter::instance().toolRadius();
    if (radius <= 0.0) radius = 0.05;

    // Parameter stores model-space radius; the tool mesh is placed in world
    // space beside the fitted model, so apply the same scale as applyFit().
    if (_fitToUnitBox && _current)
    {
        const BoundingBox bounds = BoundingBox::fromBRep(*_current);
        if (bounds.valid())
        {
            const double maxExtent =
                std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
            if (maxExtent > 0.0) radius /= maxExtent;
        }
    }

    return static_cast<float>(radius);
}

void RenderManager::rebuildTool(bool preservePose)
{
    vsg::dmat4 previousMatrix;
    const bool hadPose = _toolTransform != nullptr;
    if (hadPose) previousMatrix = _toolTransform->matrix;

    if (_toolTransform)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _toolTransform), children.end());
        _toolTransform = nullptr;
    }

    if (_toolType == ToolType::None)
    {
        if (_viewer) _viewer->request();
        return;
    }

    const float radius = worldToolRadius();
    const float height = radius * toolHeightFactor;
    const TriangleMesh mesh = createToolMesh(_toolType, radius, height);
    if (mesh.triangles.empty()) return;

    const BRep toolBRep = BRep::fromTriangles(mesh);
    const auto& verts = toolBRep.vertices();
    const auto& faceOffsets = toolBRep.faceOffsets();
    const auto& faceVertices = toolBRep.faceVertices();

    auto positions = vsg::vec3Array::create(verts.size());
    for (std::size_t i = 0; i < verts.size(); ++i) (*positions)[i] = verts[i];

    auto normals = vsg::vec3Array::create(verts.size(), vsg::vec3(0.0f, 0.0f, 0.0f));
    for (std::size_t f = 0; f < toolBRep.faceCount(); ++f)
    {
        const std::uint32_t s = faceOffsets[f];
        if (faceOffsets[f + 1] - s < 3) continue;
        const std::uint32_t ia = faceVertices[s];
        const std::uint32_t ib = faceVertices[s + 1];
        const std::uint32_t ic = faceVertices[s + 2];
        const vsg::vec3 fn = vsg::cross(verts[ib] - verts[ia], verts[ic] - verts[ia]);
        (*normals)[ia] += fn;
        (*normals)[ib] += fn;
        (*normals)[ic] += fn;
    }
    for (auto& n : *normals)
    {
        const float len = vsg::length(n);
        n = (len > 0.0f) ? n / len : vsg::vec3(0.0f, 0.0f, 1.0f);
    }

    auto colors = vsg::vec4Array::create(1, _toolColor);
    auto indices = vsg::uintArray::create(faceVertices.size());
    for (std::size_t i = 0; i < faceVertices.size(); ++i) (*indices)[i] = faceVertices[i];

    // Always a facet mesh, whatever view mode the model is in.
    auto drawable = buildDrawable(positions, normals, colors,
                                  VK_VERTEX_INPUT_RATE_INSTANCE, indices, false);

    _toolTransform = vsg::MatrixTransform::create();
    _toolTransform->addChild(drawable);

    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(_toolTransform);
        if (compileResult) vsg::updateViewer(*_viewer, compileResult);
    }

    _scene->addChild(_toolTransform);

    if (preservePose && hadPose)
        _toolTransform->matrix = previousMatrix;
    else
        // Start on the top of a unit-box model so the shank rises toward +Z
        // instead of into the camera. The tracker overwrites this on the next move.
        setToolPose(vsg::dvec3(0.0, 0.0, 0.35), vsg::dvec3(0.0, 0.15, 1.0));

    if (_viewer) _viewer->request();
}

void RenderManager::setToolPose(const vsg::dvec3& position, const vsg::dvec3& direction)
{
    if (!_toolTransform || _toolType == ToolType::None) return;

    vsg::dvec3 z = direction;
    const double zLen = vsg::length(z);
    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);
    else z /= zLen;

    // Prefer world +Z as the reference "up" when building the tool frame; fall
    // back to +X when the axis is nearly vertical.
    vsg::dvec3 up(0.0, 0.0, 1.0);
    if (std::abs(vsg::dot(z, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);

    vsg::dvec3 x = vsg::cross(up, z);
    const double xLen = vsg::length(x);
    if (xLen <= 0.0) x = vsg::dvec3(1.0, 0.0, 0.0);
    else x /= xLen;

    const vsg::dvec3 y = vsg::cross(z, x);

    _toolTransform->matrix = vsg::dmat4(x.x, x.y, x.z, 0.0,
                                        y.x, y.y, y.z, 0.0,
                                        z.x, z.y, z.z, 0.0,
                                        position.x, position.y, position.z, 1.0);

    // Always record the CPU swept volume while a tool is active. Visibility
    // of the VSG node is gated separately by _showSweptVolume.
    bool sweepChanged = false;
    if (_toolType != ToolType::None)
        sweepChanged = recordSweepStep(ToolPose{position, z});

    if (sweepChanged)
        applyBooleanToRayModel();
    else if (_viewer)
        _viewer->request();
}

void RenderManager::setSweptVolumeVisible(bool visible)
{
    if (visible == _showSweptVolume) return;
    _showSweptVolume = visible;

    if (_showSweptVolume)
        publishSweptVolume();
    else if (_sweptNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _sweptNode), children.end());
        _sweptNode = nullptr;
        if (_viewer) _viewer->request();
    }
}

void RenderManager::clearSweptVolume()
{
    if (_sweptNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _sweptNode), children.end());
        _sweptNode = nullptr;
    }

    _sweptVolume.reset();
    // Keep recording ready whenever a cutter is active.
    if (_toolType != ToolType::None) _sweptVolume = SweptVolume{};

    // Without a sweep there is nothing new to boolean against; keep any cuts
    // already applied (see applyBooleanToRayModel).
    applyBooleanToRayModel();
}

bool RenderManager::recordSweepStep(const ToolPose& pose)
{
    if (_toolType == ToolType::None) return false;

    if (!_sweptVolume) _sweptVolume = SweptVolume{};

    SweptVolume& sweep = *_sweptVolume;
    if (!sweep.lastPose())
    {
        sweep.setLastPose(pose);
        return false;
    }

    const float radius = worldToolRadius();
    const double move = vsg::length(pose.position - sweep.lastPose()->position);

    // Keep a small dead-band against mouse jitter. When a boolean op is active,
    // use a tighter threshold so the cut tracks the tool more closely.
    const double minMoveFactor =
        (Parameter::instance().booleanOp() != BooleanOp::None) ? 0.05 : 0.25;
    if (move < static_cast<double>(radius) * minMoveFactor) return false;

    const ToolPose tipA = *sweep.lastPose();

    // Last-only mode: drop the previous segment's mesh/BVH and scene node
    // before writing the new one, so only the latest sweep is kept.
    if (Parameter::instance().showLastSweptVolumeOnly())
    {
        if (_sweptNode)
        {
            auto& children = _scene->children;
            children.erase(std::remove(children.begin(), children.end(), _sweptNode),
                           children.end());
            _sweptNode = nullptr;
        }
        sweep.clearGeometry();
    }

    sweep.appendSegment(_toolType, radius, radius * toolHeightFactor, tipA, pose);
    publishSweptVolume();
    return true;
}

void RenderManager::publishSweptVolume()
{
    if (_sweptNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _sweptNode), children.end());
        _sweptNode = nullptr;
    }

    // Checkbox off: keep the CPU mesh / BVH for boolean, but do not draw it.
    if (!_showSweptVolume) return;
    if (!_sweptVolume || _sweptVolume->empty()) return;

    const TriangleMesh& mesh = _sweptVolume->mesh();
    const std::size_t triCount = mesh.triangles.size();
    auto positions = vsg::vec3Array::create(triCount * 3);
    auto normals = vsg::vec3Array::create(triCount * 3);
    auto indices = vsg::uintArray::create(triCount * 3);

    for (std::size_t i = 0; i < triCount; ++i)
    {
        const MeshTriangle& tri = mesh.triangles[i];
        const std::size_t base = i * 3;
        (*positions)[base] = tri.v0;
        (*positions)[base + 1] = tri.v1;
        (*positions)[base + 2] = tri.v2;
        (*normals)[base] = tri.normal;
        (*normals)[base + 1] = tri.normal;
        (*normals)[base + 2] = tri.normal;
        (*indices)[base] = static_cast<uint32_t>(base);
        (*indices)[base + 1] = static_cast<uint32_t>(base + 1);
        (*indices)[base + 2] = static_cast<uint32_t>(base + 2);
    }

    auto colors = vsg::vec4Array::create(1, _sweptColor);
    _sweptNode = buildDrawable(positions, normals, colors,
                               VK_VERTEX_INPUT_RATE_INSTANCE, indices,
                               false, true);

    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(_sweptNode);
        if (compileResult) vsg::updateViewer(*_viewer, compileResult);
    }

    _scene->addChild(_sweptNode);
}

void RenderManager::setBooleanOp(BooleanOp op)
{
    Parameter::instance().setBooleanOp(op);

    if (op == BooleanOp::None)
    {
        // Keep whatever RayModel is on screen; only stop applying new cuts.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
        if (_viewer) _viewer->request();
        return;
    }

    applyBooleanToRayModel();
}

void RenderManager::applyBooleanToRayModel()
{
    joinRayCleanup();

    if (!_sourceRayModel)
    {
        _booleanRayModel.reset();
        _rayModel = nullptr;
        return;
    }

    const BooleanOp op = Parameter::instance().booleanOp();
    if (op == BooleanOp::None)
    {
        // Do not revert to the original cast; leave the last result in place.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
    }
    else if (!_sweptVolume || _sweptVolume->empty())
    {
        // Keep any cuts already applied; only the cutter mesh went away.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
    }
    else
    {
        // Cumulative: each new sweep segment cuts the last boolean result.
        const RayModel& input = _booleanRayModel ? *_booleanRayModel : *_sourceRayModel;
        const BoundingBox bounds = input.bounds();
        RayModel next = input.withBoolean(*_sweptVolume, op, fitMatrix(bounds));
        _booleanRayModel = std::move(next);
        _rayModel = &*_booleanRayModel;
    }

    if (usesRayModel(_viewMode)) rebuild();
    else if (_viewer) _viewer->request();
}

bool RenderManager::pickToolPlacement(const vsg::Camera& camera, int32_t x, int32_t y,
                                      vsg::dvec3& position, vsg::dvec3& normal) const
{
    // Reconstruct the world-space ray the same way VSG's LineSegmentIntersector
    // does, then map it into model space so the BRep can be tested without
    // depending on the current view mode's drawables.
    const auto viewport = camera.getViewport();
    vsg::vec2 ndc(0.0f, 0.0f);
    if (viewport.width > 0 && viewport.height > 0)
    {
        ndc.set((static_cast<float>(x) - viewport.x) / viewport.width,
                (static_cast<float>(y) - viewport.y) / viewport.height);
    }

    const vsg::dmat4 projectionMatrix = camera.projectionMatrix->transform();
    const vsg::dmat4 viewMatrix = camera.viewMatrix->transform();
    const bool reverseDepth = projectionMatrix(2, 2) > 0.0;

    const vsg::dvec3 ndcNear(ndc.x * 2.0 - 1.0, ndc.y * 2.0 - 1.0,
                             reverseDepth ? viewport.maxDepth : viewport.minDepth);
    const vsg::dvec3 ndcFar(ndc.x * 2.0 - 1.0, ndc.y * 2.0 - 1.0,
                            reverseDepth ? viewport.minDepth : viewport.maxDepth);

    const vsg::dmat4 invProjection = vsg::inverse(projectionMatrix);
    const vsg::dmat4 eyeToWorld = vsg::inverse(viewMatrix);
    const vsg::dvec3 worldNear = eyeToWorld * (invProjection * ndcNear);
    const vsg::dvec3 worldFar = eyeToWorld * (invProjection * ndcFar);
    const vsg::dvec3 worldDir = worldFar - worldNear;

    // Camera look axis in world space (VSG eye space looks down -Z). The miss
    // fallback is the plane through the focus point perpendicular to this axis,
    // i.e. the view projection plane.
    const vsg::dvec4 lookEye(0.0, 0.0, -1.0, 0.0);
    const vsg::dvec4 lookWorld = eyeToWorld * lookEye;
    vsg::dvec3 viewAxis(lookWorld.x, lookWorld.y, lookWorld.z);
    const double viewLen = vsg::length(viewAxis);
    if (viewLen > 0.0) viewAxis /= viewLen;
    else viewAxis = vsg::normalize(worldDir);

    auto projectToViewPlane = [&](const vsg::dvec3& focus) -> bool {
        const auto plane = pickPlane(worldNear, worldDir, focus, viewAxis);
        if (!plane) return false;
        position = plane->position;
        normal = plane->normal;
        return true;
    };

    if (!_current)
    {
        // No model: use the look-at target if present, otherwise the origin.
        vsg::dvec3 focus(0.0, 0.0, 0.0);
        if (auto lookAt = camera.viewMatrix.cast<vsg::LookAt>())
            focus = lookAt->center;
        return projectToViewPlane(focus);
    }

    const BoundingBox bounds = BoundingBox::fromBRep(*_current);
    const vsg::dmat4 modelToWorld = fitMatrix(bounds);
    const vsg::dmat4 worldToModel = vsg::inverse(modelToWorld);

    const vsg::dvec3 modelOrigin = worldToModel * worldNear;
    const vsg::dvec3 modelFarPt = worldToModel * worldFar;
    const vsg::dvec3 modelDir = modelFarPt - modelOrigin;

    if (const auto hit = pickBRep(*_current, modelOrigin, modelDir))
    {
        // Fitting is uniform scale + translate, so directions are unchanged.
        position = modelToWorld * hit->position;
        normal = hit->normal;
        return true;
    }

    // Miss: slide the tool on the view plane through the model centre so it
    // stays under the cursor without jumping to a fixed world Z.
    const Point3d centre = bounds.centre();
    const vsg::dvec3 focus = modelToWorld * vsg::dvec3(centre[0], centre[1], centre[2]);
    return projectToViewPlane(focus);
}

void RenderManager::attach(vsg::ref_ptr<vsg::Node> node, bool replaceExisting)
{
    if (!node) return;

    // The viewer is already running, so the new subgraph has to be compiled
    // before it can be recorded. compileManager is set up by Viewer::compile().
    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(node);
        if (compileResult) vsg::updateViewer(*_viewer, compileResult);
    }

    if (replaceExisting)
    {
        auto& children = _scene->children;
        if (_modelNode)
        {
            children.erase(std::remove(children.begin(), children.end(), _modelNode),
                           children.end());
        }
        _modelNode = node;
    }

    _scene->addChild(node);

    if (_viewer) _viewer->request();
}

} // namespace app
