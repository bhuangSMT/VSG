#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
#include "TriangleMesh.h"

namespace app
{
namespace
{

using ProfileClock = std::chrono::steady_clock;

double millisSince(ProfileClock::time_point start)
{
    const std::chrono::duration<double, std::milli> elapsed = ProfileClock::now() - start;
    return elapsed.count();
}

// Cells the boolean will visit for this dirty box, summed over present axes.
// This is the work the sweep AABB actually buys, independent of what it cuts.
std::size_t dirtyWindowCells(const RayModel& model, const BoundingBox& modelAabb)
{
    if (!modelAabb.valid()) return 0;

    std::size_t cells = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = model.grid(axis);
        if (!grid) continue;

        std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
        if (!gridWindowFromModelAabb(*grid, modelAabb, iu0, iu1, iv0, iv1)) continue;
        cells += static_cast<std::size_t>(iu1 - iu0 + 1) *
                 static_cast<std::size_t>(iv1 - iv0 + 1);
    }
    return cells;
}

void transformTriangleMesh(TriangleMesh& mesh, const vsg::dmat4& matrix)
{
    for (MeshTriangle& tri : mesh.triangles)
    {
        const vsg::dvec3 a = matrix * vsg::dvec3(tri.v0.x, tri.v0.y, tri.v0.z);
        const vsg::dvec3 b = matrix * vsg::dvec3(tri.v1.x, tri.v1.y, tri.v1.z);
        const vsg::dvec3 c = matrix * vsg::dvec3(tri.v2.x, tri.v2.y, tri.v2.z);
        vsg::dvec3 n = vsg::cross(b - a, c - a);
        const double len = vsg::length(n);
        if (len > 1.0e-18) n /= len;
        else n = vsg::dvec3(0.0, 0.0, 1.0);

        tri.v0 = vsg::vec3(static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z));
        tri.v1 = vsg::vec3(static_cast<float>(b.x), static_cast<float>(b.y), static_cast<float>(b.z));
        tri.v2 = vsg::vec3(static_cast<float>(c.x), static_cast<float>(c.y), static_cast<float>(c.z));
        tri.normal = vsg::vec3(static_cast<float>(n.x), static_cast<float>(n.y), static_cast<float>(n.z));
    }
}

// Live intervals vs entries actually held by the per-axis interval pools. The
// ratio is what IntervalPool::compact keeps bounded. ("slots" is a Qt macro.)
void poolOccupancy(const RayModel& model, std::size_t& live, std::size_t& reserved)
{
    live = 0;
    reserved = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = model.grid(axis);
        if (!grid) continue;
        live += grid->intervalCount();
        reserved += grid->pool.data.size();
    }
}

} // namespace

RenderManager::RenderManager(vsg::ref_ptr<vsgQt::Viewer> viewer,
                             vsg::ref_ptr<vsg::Group> scene,
                             vsg::ref_ptr<vsg::Options> options) :
    _viewer(viewer),
    _scene(scene),
    _options(options)
{
    if (!_scene) throw std::runtime_error("RenderManager requires a valid scene root.");
}

RenderManager::~RenderManager() = default;

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

    const int stride = rayModel.strideForRayBudget(maxRenderedRays);
    const std::size_t rays = rayModel.rayCountAtStride(stride);
    const std::size_t pointCount = rays * 2;

    auto positions = vsg::vec3Array::create(pointCount);
    auto normals = vsg::vec3Array::create(pointCount, vsg::vec3(0.0f, 0.0f, 1.0f));
    auto colors = vsg::vec4Array::create(pointCount);
    auto indices = vsg::uintArray::create(pointCount);

    struct SpanWrite
    {
        std::size_t axis = 0;
        std::uint32_t iu = 0;
        std::uint32_t iv = 0;
        const RayGrid* grid = nullptr;
        const Interval* intervals = nullptr;
        std::uint32_t intervalCount = 0;
        std::size_t pointOffset = 0;
    };

    std::vector<SpanWrite> writes;
    writes.reserve(rays);
    std::size_t next = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid) continue;

        for (std::uint32_t iv = 0; iv < grid->height; ++iv)
        {
            if (static_cast<int>(iv) % stride != 0) continue;
            for (std::uint32_t iu = 0; iu < grid->width; ++iu)
            {
                if (static_cast<int>(iu) % stride != 0) continue;
                const RaySlot& slot = grid->at(iu, iv);
                if (slot.empty()) continue;
                auto spans = grid->pool.span(slot);
                writes.push_back(SpanWrite{axis, iu, iv, grid, spans.data(),
                                           slot.intervalCount, next});
                next += static_cast<std::size_t>(slot.intervalCount) * 2;
            }
        }
    }

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, writes.size()),
        [&](const tbb::blocked_range<std::size_t>& range) {
            for (std::size_t i = range.begin(); i != range.end(); ++i)
            {
                const SpanWrite& write = writes[i];
                const RayGrid& grid = *write.grid;
                const std::size_t axis = write.axis;
                const std::size_t u = (axis + 1) % 3;
                const std::size_t v = (axis + 2) % 3;
                const double u0 = grid.sampleU(write.iu);
                const double v0 = grid.sampleV(write.iv);
                const vsg::vec4& axisColor = _rayColors[axis];
                std::size_t point = write.pointOffset;

                for (std::uint32_t s = 0; s < write.intervalCount; ++s)
                {
                    const Interval& iv = write.intervals[s];
                    const vsg::vec4& color = iv.fromBoolean() ? _toolColor : axisColor;

                    Point3d start{0.0, 0.0, 0.0};
                    Point3d end{0.0, 0.0, 0.0};
                    start[axis] = grid.fromTick(iv.begin);
                    end[axis] = grid.fromTick(iv.end);
                    start[u] = u0;
                    end[u] = u0;
                    start[v] = v0;
                    end[v] = v0;

                    (*positions)[point] = vsg::vec3(static_cast<float>(start[0]),
                                                    static_cast<float>(start[1]),
                                                    static_cast<float>(start[2]));
                    (*colors)[point] = color;
                    ++point;
                    (*positions)[point] = vsg::vec3(static_cast<float>(end[0]),
                                                    static_cast<float>(end[1]),
                                                    static_cast<float>(end[2]));
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

std::array<float, 3> RenderManager::splatRadii(const RayModel& rayModel, int stride) const
{
    const float s = static_cast<float>(stride < 1 ? 1 : stride);
    return {splatRadius(rayModel, 0) * s, splatRadius(rayModel, 1) * s,
            splatRadius(rayModel, 2) * s};
}

SplatStyle RenderManager::splatStyle() const
{
    SplatStyle style;
    style.stockColor = _splatColor;
    style.toolColor = _toolColor;
    style.opacity = _splatOpacity;
    return style;
}

void RenderManager::rebuildSplatCache()
{
    if (!_rayModel) return;

    const int stride = _rayModel->strideForRayBudget(maxRenderedRays);
    auto drawable =
        _splatCache.rebuild(*_rayModel, stride, splatRadii(*_rayModel, stride), splatStyle());
    attach(applyFit(drawable, _rayModel->bounds()), true);
}

vsg::ref_ptr<vsg::Node> RenderManager::createSplatNode(const RayModel& rayModel) const
{
    auto chainLock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    const int stride = rayModel.strideForRayBudget(maxRenderedRays);

    std::vector<Splat> splats;
    splats.reserve(rayModel.rayCountAtStride(stride) * 2);

    vsg::vec4 color = _splatColor;
    color.a = _splatOpacity;

    vsg::vec4 toolColor = _toolColor;
    toolColor.a = _splatOpacity;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid) continue;

        const float radius = splatRadius(rayModel, axis) * static_cast<float>(stride);
        const std::size_t u = (axis + 1) % 3;
        const std::size_t v = (axis + 2) % 3;

        for (std::uint32_t iv = 0; iv < grid->height; ++iv)
        {
            if (static_cast<int>(iv) % stride != 0) continue;
            for (std::uint32_t iu = 0; iu < grid->width; ++iu)
            {
                if (static_cast<int>(iu) % stride != 0) continue;
                const RaySlot& slot = grid->at(iu, iv);
                if (slot.empty()) continue;

                const double u0 = grid->sampleU(iu);
                const double v0 = grid->sampleV(iv);
                auto spans = grid->pool.span(slot);

                for (const Interval& span : spans)
                {
                    Point3d start{0.0, 0.0, 0.0};
                    Point3d end{0.0, 0.0, 0.0};
                    start[axis] = grid->fromTick(span.begin);
                    end[axis] = grid->fromTick(span.end);
                    start[u] = u0;
                    end[u] = u0;
                    start[v] = v0;
                    end[v] = v0;

                    auto normalOrAxis = [axis](const Normal3f& n, bool enter) {
                        const float len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
                        if (len2 < 1.0e-12f)
                        {
                            vsg::vec3 fallback(0.0f, 0.0f, 0.0f);
                            fallback[static_cast<uint32_t>(axis)] = enter ? -1.0f : 1.0f;
                            return fallback;
                        }
                        return vsg::vec3(n[0], n[1], n[2]);
                    };

                    const vsg::vec4& splatColor = span.fromBoolean() ? toolColor : color;
                    splats.push_back({vsg::vec3(static_cast<float>(start[0]),
                                                static_cast<float>(start[1]),
                                                static_cast<float>(start[2])),
                                      normalOrAxis(span.beginNormal, true), splatColor, radius});
                    splats.push_back({vsg::vec3(static_cast<float>(end[0]),
                                                static_cast<float>(end[1]),
                                                static_cast<float>(end[2])),
                                      normalOrAxis(span.endNormal, false), splatColor, radius});
                }
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
    const Point3d resolution = model.resolution();
    const std::size_t rays = model.rayCount();

    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();

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
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
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
    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
    _rayModels.clear();
    _rayModelOrder.clear();
    _cachedRays = 0;
    _splatCache.clear();
}

void RenderManager::setViewMode(ViewMode mode)
{
    if (mode == _viewMode) return;
    _viewMode = mode;
    rebuild();
}

void RenderManager::rebuild()
{
    // The ray modes need a model to draw; without one fall back to the surface
    // so the viewport never goes blank.
    if (usesRayModel(_viewMode) && _rayModel && _rayModel->rayCount() > 0)
    {
        if (_viewMode == ViewMode::RayGS)
        {
            rebuildSplatCache();
            return;
        }

        attach(createRayNode(*_rayModel), true);
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
    _cutSweep.reset();
    _lastToolPose.reset();
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
        _lastToolPose.reset();
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

float RenderManager::worldToolLength() const
{
    double length = Parameter::instance().toolLength();
    if (length <= 0.0) length = static_cast<double>(worldToolRadius()) * 2.8;

    if (_fitToUnitBox && _current)
    {
        const BoundingBox bounds = BoundingBox::fromBRep(*_current);
        if (bounds.valid())
        {
            const double maxExtent =
                std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
            if (maxExtent > 0.0) length /= maxExtent;
        }
    }

    return static_cast<float>(length);
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
    const float height = worldToolLength();
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

    // Inspection replaces the solid cutter with the wireframe swept volume.
    setToolNodeAttached(Parameter::instance().booleanOp() != BooleanOp::Inspection);

    if (preservePose && hadPose)
    {
        _toolTransform->matrix = previousMatrix;
        if (Parameter::instance().booleanOp() == BooleanOp::Inspection && _lastToolPose)
        {
            placeInspectionCutter(*_lastToolPose, true);
            applyBooleanToRayModel();
        }
    }
    else
        // Start on the top of a unit-box model so the shank rises toward +Z
        // instead of into the camera. The tracker overwrites this on the next move.
        setToolPose(vsg::dvec3(0.0, 0.0, 0.35), vsg::dvec3(0.0, 0.15, 1.0));

    if (_viewer) _viewer->request();
}

void RenderManager::setToolNodeAttached(bool attached)
{
    if (!_toolTransform) return;

    auto& children = _scene->children;
    const auto found = std::find(children.begin(), children.end(), _toolTransform);
    if (attached)
    {
        if (found == children.end()) _scene->addChild(_toolTransform);
    }
    else if (found != children.end())
    {
        children.erase(found);
    }
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

    // Mesh / sweep frames put the tip at the origin and the sphere centre at
    // +radius along the tool axis. For ball nose and sphere, the mouse hit is
    // that centre, so shift the tip down by radius along -z.
    vsg::dvec3 tip = position;
    if (_toolType == ToolType::BallNose || _toolType == ToolType::Sphere)
        tip = position - z * static_cast<double>(worldToolRadius());

    _toolTransform->matrix = vsg::dmat4(x.x, x.y, x.z, 0.0,
                                        y.x, y.y, y.z, 0.0,
                                        z.x, z.y, z.z, 0.0,
                                        tip.x, tip.y, tip.z, 1.0);

    const ToolPose pose{tip, z};
    _lastToolPose = pose;

    if (Parameter::instance().booleanOp() == BooleanOp::Inspection)
    {
        const bool movedEnough = placeInspectionCutter(pose, false);
        if (movedEnough)
            applyBooleanToRayModel();
        else if (_viewer)
            _viewer->request();
        return;
    }

    // Always record the CPU swept volume while a tool is active. Visibility
    // of the VSG node is gated separately by _showSweptVolume.
    bool sweepChanged = false;
    if (_toolType != ToolType::None)
        sweepChanged = recordSweepStep(pose);

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
    _cutSweep.reset();
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

    // Boolean always uses this one segment. Stock (_booleanRayModel) already
    // holds prior cuts; re-walking the whole path would only grow cost.
    SweptVolume step;
    step.appendSegment(_toolType, radius, worldToolLength(), tipA, pose);
    if (step.empty())
    {
        sweep.setLastPose(pose);
        return false;
    }
    _cutSweep = step;

    if (Parameter::instance().showLastSweptVolumeOnly())
    {
        if (_sweptNode)
        {
            auto& children = _scene->children;
            children.erase(std::remove(children.begin(), children.end(), _sweptNode),
                           children.end());
            _sweptNode = nullptr;
        }
        sweep = std::move(step);
    }
    else
    {
        // Keep the path for drawing only; skip BVH on the accumulator.
        sweep.appendTriangles(_cutSweep->mesh(), false);
        sweep.setLastPose(pose);
    }

    publishSweptVolume();
    return true;
}

bool RenderManager::placeInspectionCutter(const ToolPose& pose, bool forceBoolean)
{
    if (_toolType == ToolType::None || !_toolTransform) return false;

    TriangleMesh mesh = createToolMesh(_toolType, worldToolRadius(), worldToolLength());
    if (mesh.triangles.empty()) return false;

    transformTriangleMesh(mesh, _toolTransform->matrix);

    SweptVolume cutter;
    cutter.appendTriangles(mesh, true);
    cutter.setLastPose(pose);
    if (cutter.empty()) return false;

    _cutSweep = std::move(cutter);

    // Keep the path accumulator's tip in step so leaving Inspection does not
    // boolean a jump from the last real cut to the current mouse pose.
    if (!_sweptVolume) _sweptVolume = SweptVolume{};
    _sweptVolume->setLastPose(pose);

    publishSweptVolume();

    if (!forceBoolean && _inspectionBooleanPose)
    {
        const float radius = worldToolRadius();
        const double move = vsg::length(pose.position - _inspectionBooleanPose->position);
        if (move < static_cast<double>(radius) * 0.05) return false;
    }

    _inspectionBooleanPose = pose;
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

    // Inspection draws the cutter at the current pose; the other modes draw
    // the accumulated path.
    const bool inspection = Parameter::instance().booleanOp() == BooleanOp::Inspection;
    const SweptVolume* drawn = nullptr;
    if (inspection)
    {
        if (_cutSweep && !_cutSweep->empty()) drawn = &*_cutSweep;
    }
    else if (_sweptVolume && !_sweptVolume->empty())
    {
        drawn = &*_sweptVolume;
    }
    if (!drawn) return;

    const TriangleMesh& mesh = drawn->mesh();
    const auto triCount = mesh.triangles.size();

    auto positions = vsg::vec3Array::create(triCount * 3);
    auto normals = vsg::vec3Array::create(triCount * 3);

    for (auto i = decltype(triCount){0}; i < triCount; ++i)
    {
        const MeshTriangle& tri = mesh.triangles[i];
        const auto base = i * 3;
        (*positions)[base] = tri.v0;
        (*positions)[base + 1] = tri.v1;
        (*positions)[base + 2] = tri.v2;
        (*normals)[base] = tri.normal;
        (*normals)[base + 1] = tri.normal;
        (*normals)[base + 2] = tri.normal;
    }

    if (inspection)
    {
        auto indices = vsg::uintArray::create(triCount * 6);
        for (auto i = decltype(triCount){0}; i < triCount; ++i)
        {
            const auto base = static_cast<uint32_t>(i * 3);
            const auto edge = i * 6;
            (*indices)[edge] = base;
            (*indices)[edge + 1] = base + 1;
            (*indices)[edge + 2] = base + 1;
            (*indices)[edge + 3] = base + 2;
            (*indices)[edge + 4] = base + 2;
            (*indices)[edge + 5] = base;
        }

        auto colors = vsg::vec4Array::create(1, _wireframeColor);
        _sweptNode = buildDrawable(positions, normals, colors,
                                   VK_VERTEX_INPUT_RATE_INSTANCE, indices,
                                   true);
    }
    else
    {
        auto indices = vsg::uintArray::create(triCount * 3);
        for (auto i = decltype(triCount){0}; i < triCount; ++i)
        {
            const auto base = i * 3;
            (*indices)[base] = static_cast<uint32_t>(base);
            (*indices)[base + 1] = static_cast<uint32_t>(base + 1);
            (*indices)[base + 2] = static_cast<uint32_t>(base + 2);
        }

        auto colors = vsg::vec4Array::create(1, _sweptColor);
        _sweptNode = buildDrawable(positions, normals, colors,
                                   VK_VERTEX_INPUT_RATE_INSTANCE, indices,
                                   false, true);
    }

    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(_sweptNode);
        if (compileResult) vsg::updateViewer(*_viewer, compileResult);
    }

    _scene->addChild(_sweptNode);
}

void RenderManager::setBooleanOp(BooleanOp op)
{
    const BooleanOp previous = Parameter::instance().booleanOp();
    Parameter::instance().setBooleanOp(op);

    if (previous == BooleanOp::Inspection && op != BooleanOp::Inspection)
    {
        // Drop the preview hole and put accumulated cuts (if any) back on
        // screen. The inspection cutter is not a real cut.
        _inspectionRayModel.reset();
        _inspectionPrevAabb = {};
        _inspectionBooleanPose.reset();
        _cutSweep.reset();
        publishSweptVolume();
        setToolNodeAttached(true);
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
        if (usesRayModel(_viewMode) && _rayModel)
            rebuild();
        else if (_viewer)
            _viewer->request();
        return;
    }

    if (op == BooleanOp::Inspection)
    {
        _cutSweep.reset();
        if (_toolTransform && _toolType != ToolType::None && _lastToolPose)
            placeInspectionCutter(*_lastToolPose, true);
        setToolNodeAttached(false);
        applyBooleanToRayModel();
        return;
    }

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
    if (!_sourceRayModel)
    {
        _booleanRayModel.reset();
        _inspectionRayModel.reset();
        _inspectionPrevAabb = {};
        _rayModel = nullptr;
        _splatCache.clear();
        return;
    }

    const BooleanOp op = Parameter::instance().booleanOp();
    BoundingBox dirtyModelAabb;
    BoundingBox restoreAabb;
    bool haveDirtyRegion = false;
    double booleanMs = 0.0;
    bool raysMutated = false;

    // Tool motion calls this on every sweep step, so the branches below that
    // only re-point _rayModel must not trigger a redraw of unchanged rays.
    const RayModel* const displayedBefore = _rayModel;
    const bool wasInspecting =
        _inspectionRayModel.has_value() && displayedBefore == &*_inspectionRayModel;
    const bool showingSource = displayedBefore == _sourceRayModel;

    if (op == BooleanOp::None)
    {
        // Do not revert to the original cast; leave the last result in place.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
    }
    else if (op == BooleanOp::Inspection)
    {
        if (!_cutSweep || _cutSweep->empty())
        {
            _rayModel = nullptr;
            _inspectionRayModel.reset();
            _inspectionPrevAabb = {};
            _rayModel = _sourceRayModel;
            raysMutated = !showingSource;
        }
        else
        {
            // Fresh stock every move: clone the cached original, then subtract
            // the cutter at this pose. Do not touch _booleanRayModel.
            _rayModel = nullptr;
            _inspectionRayModel = _sourceRayModel->clone();

            const BoundingBox bounds = _inspectionRayModel->bounds();
            const vsg::dmat4 modelToWorld = fitMatrix(bounds);
            const vsg::dmat4 worldToModel = vsg::inverse(modelToWorld);

            BoundingBox currentDirty;
            if (_cutSweep->bvh().bounds().valid())
                currentDirty = modelAabbFromWorld(_cutSweep->bvh().bounds(), worldToModel);

            const auto booleanStart = ProfileClock::now();
            _inspectionRayModel->booleanInPlace(*_cutSweep, BooleanOp::Subtraction,
                                                modelToWorld);
            booleanMs = millisSince(booleanStart);
            _rayModel = &*_inspectionRayModel;
            raysMutated = true;

            dirtyModelAabb = currentDirty;
            if (wasInspecting) restoreAabb = _inspectionPrevAabb;
            // Patch when the previous display was already original stock
            // (plus at most the last preview hole). Accumulated cuts need a
            // full rebuild so old holes do not linger in the splat cache.
            haveDirtyRegion = currentDirty.valid() && (wasInspecting || showingSource);
            _inspectionPrevAabb = currentDirty;
        }
    }
    else if (!_cutSweep || _cutSweep->empty())
    {
        // Keep any cuts already applied; only the cutter mesh went away.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
    }
    else
    {
        // Cumulative stock: mutate a working copy in place. Fork from source once.
        // Each call subtracts/unions only the newest segment (_cutSweep).
        if (!_booleanRayModel)
            _booleanRayModel = _sourceRayModel->clone();

        const BoundingBox bounds = _booleanRayModel->bounds();
        const vsg::dmat4 modelToWorld = fitMatrix(bounds);
        const vsg::dmat4 worldToModel = vsg::inverse(modelToWorld);

        if (_cutSweep->bvh().bounds().valid())
        {
            dirtyModelAabb =
                modelAabbFromWorld(_cutSweep->bvh().bounds(), worldToModel);
            haveDirtyRegion = dirtyModelAabb.valid();
        }

        const auto booleanStart = ProfileClock::now();
        _booleanRayModel->booleanInPlace(*_cutSweep, op, modelToWorld);
        booleanMs = millisSince(booleanStart);
        _rayModel = &*_booleanRayModel;
        raysMutated = true;
    }

    if (!raysMutated && _rayModel == displayedBefore)
    {
        // No cut ran and the same model is still on screen: the tool (and its
        // swept volume) moved, but the rays did not. A redraw is enough, and a
        // full Ray-GS rebuild here would cost more than the frame itself.
        if (_viewer) _viewer->request();
        return;
    }

    if (_viewMode == ViewMode::RayGS && _rayModel && haveDirtyRegion && !_splatCache.empty())
    {
        const int stride = _rayModel->strideForRayBudget(maxRenderedRays);
        const auto patchStart = ProfileClock::now();
        const auto radii = splatRadii(*_rayModel, stride);
        const SplatStyle style = splatStyle();
        PatchResult patched = PatchResult::Ok;
        if (restoreAabb.valid())
        {
            patched = _splatCache.updateRegion(*_rayModel, restoreAabb, stride, radii, style);
            if (patched == PatchResult::Ok && dirtyModelAabb.valid())
                patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride,
                                                   radii, style);
        }
        else
        {
            patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride,
                                               radii, style);
        }
        if (patched == PatchResult::Ok)
        {
            logCutProfile(booleanMs, "patch", millisSince(patchStart), dirtyModelAabb);
            if (_viewer) _viewer->request();
            return;
        }
        // updateRegion bailed part way through; the rebuild below repacks it.
        if (_profiling)
            std::printf("  splat patch failed (%s) after %.2f ms\n",
                        toString(patched), millisSince(patchStart));
    }

    if (usesRayModel(_viewMode))
    {
        const auto rebuildStart = ProfileClock::now();
        rebuild();
        logCutProfile(booleanMs, "rebuild", millisSince(rebuildStart), dirtyModelAabb);
    }
    else
    {
        logCutProfile(booleanMs, "no-draw", 0.0, dirtyModelAabb);
        if (_viewer) _viewer->request();
    }
}

void RenderManager::logCutProfile(double booleanMs, const char* drawPath, double drawMs,
                                  const BoundingBox& dirtyModelAabb)
{
    if (!_profiling) return;

    ++_cutIndex;

    std::size_t liveIntervals = 0;
    std::size_t poolReserved = 0;
    if (_rayModel) poolOccupancy(*_rayModel, liveIntervals, poolReserved);

    const double poolRatio =
        liveIntervals > 0 ? static_cast<double>(poolReserved) / static_cast<double>(liveIntervals)
                          : 0.0;
    const std::size_t splatLive = _splatCache.liveEndpoints();
    const std::size_t splatCap = _splatCache.capacity();
    const double splatFill =
        splatCap > 0 ? static_cast<double>(splatLive) / static_cast<double>(splatCap) : 0.0;

    const std::size_t windowCells =
        _rayModel ? dirtyWindowCells(*_rayModel, dirtyModelAabb) : 0;
    const std::size_t sweepTris = _cutSweep ? _cutSweep->mesh().triangles.size() : 0;

    std::printf("cut %-4lld total %7.2f ms  boolean %7.2f ms  %-7s %6.2f ms"
                "  window %9zu cells  sweep %6zu tris"
                "  intervals %8zu  pool x%.2f  splat %8zu/%-8zu %3.0f%%\n",
                _cutIndex, booleanMs + drawMs, booleanMs, drawPath, drawMs,
                windowCells, sweepTris,
                liveIntervals, poolRatio, splatLive, splatCap, splatFill * 100.0);
    std::fflush(stdout);
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
