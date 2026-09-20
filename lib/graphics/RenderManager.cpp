#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <QTimer>

#include <vsg/vk/Device.h>
#include <vsg/vk/PhysicalDevice.h>

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

bool accumulatesCutMesh(BooleanOp op)
{
    return op == BooleanOp::Subtraction || op == BooleanOp::Union;
}

bool cutMeshEnabled()
{
    return Parameter::instance().cutMeshDisplay();
}

bool skipCutSplatEnds(bool /*cutFaceLive*/)
{
    // Skip interior cut-tagged Gaussians only while the orange mesh overlay is
    // the intended cut surface. Display off → keep cut disks.
    return cutMeshEnabled();
}

// Sub-cell overlap so consecutive closed sweeps are not coincident at the
// shared pose. Stays below one grid tick and below a quarter of the segment.
double backExtendEpsilon(const RayModel* model, double radius, double segmentLen)
{
    double eps = std::max(1.0e-6, radius * 1.0e-3);
    if (model)
    {
        double unit = 0.0;
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const RayGrid* g = model->grid(axis);
            if (!g || !(g->unit > 0.0f)) continue;
            const double u = static_cast<double>(g->unit);
            if (unit <= 0.0 || u < unit) unit = u;
        }
        if (unit > 0.0) eps = std::min(eps, 0.25 * unit);
    }
    if (segmentLen > 0.0) eps = std::min(eps, 0.25 * segmentLen);
    return eps;
}

// Patch / rebuild cut-face GPU overlay only while Cut mesh display is on.
// Off means cut-tagged splat disks only — skip remesh entirely.
void syncCutFaceOverlay(GaussianSplatCache& cache, const RayModel& rayModel, int cutStride,
                        const vsg::vec4& color, BooleanOp op, const BoundingBox& dirtyModelAabb,
                        bool allowPatch)
{
    if (!cutMeshEnabled()) return;

    if (accumulatesCutMesh(op))
    {
        if (allowPatch && dirtyModelAabb.valid() && cache.hasCutFace())
            cache.patchCutFace(rayModel, cutStride, dirtyModelAabb, color);
        else
            cache.rebuildCutFace(rayModel, cutStride, color);
    }
    else if (!cache.hasCutFace())
        cache.rebuildCutFace(rayModel, cutStride, color);
    else
        cache.showCutFace();
}

BoundingBox intersectAabb(const BoundingBox& a, const BoundingBox& b)
{
    if (!a.valid() || !b.valid()) return {};
    Point3d mn{std::max(a.min()[0], b.min()[0]), std::max(a.min()[1], b.min()[1]),
               std::max(a.min()[2], b.min()[2])};
    Point3d mx{std::min(a.max()[0], b.max()[0]), std::min(a.max()[1], b.max()[1]),
               std::min(a.max()[2], b.max()[2])};
    if (mn[0] > mx[0] || mn[1] > mx[1] || mn[2] > mx[2]) return {};
    return BoundingBox(mn, mx);
}

BoundingBox expandAabb(const BoundingBox& box, double radius)
{
    if (!box.valid()) return {};
    Point3d mn = box.min();
    Point3d mx = box.max();
    mn[0] -= radius;
    mn[1] -= radius;
    mn[2] -= radius;
    mx[0] += radius;
    mx[1] += radius;
    mx[2] += radius;
    return BoundingBox(mn, mx);
}

// Far-slab hit along dir. 0 when the ray misses or the box is behind.
double rayAabbExitT(const vsg::dvec3& origin, const vsg::dvec3& dir, const BoundingBox& box)
{
    if (!box.valid()) return 0.0;

    double tEnter = -1.0e300;
    double tExit = 1.0e300;
    const Point3d& mn = box.min();
    const Point3d& mx = box.max();
    for (int i = 0; i < 3; ++i)
    {
        const double o = origin[i];
        const double d = dir[i];
        const double a = mn[i];
        const double b = mx[i];
        if (std::abs(d) < 1.0e-12)
        {
            if (o < a || o > b) return 0.0;
            continue;
        }
        double t0 = (a - o) / d;
        double t1 = (b - o) / d;
        if (t0 > t1) std::swap(t0, t1);
        if (t0 > tEnter) tEnter = t0;
        if (t1 < tExit) tExit = t1;
        if (tEnter > tExit) return 0.0;
    }
    if (tExit < 0.0) return 0.0;
    return tExit;
}

double millisSince(ProfileClock::time_point start)
{
    const std::chrono::duration<double, std::milli> elapsed = ProfileClock::now() - start;
    return elapsed.count();
}

// Cells the boolean will visit for this dirty box, summed over present axes.
// This is the work the sweep AABB actually buys, independent of what it cuts.
[[maybe_unused]] std::size_t dirtyWindowCells(const RayModel& model, const BoundingBox& modelAabb)
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
    showWorldAxes(true);
}

RenderManager::~RenderManager()
{
    if (_splatViewDebounce)
    {
        _splatViewDebounce->stop();
        delete _splatViewDebounce;
        _splatViewDebounce = nullptr;
    }
}

void RenderManager::configureRayBudgets(vsg::ref_ptr<vsg::Device> device)
{
    constexpr std::size_t kBytesPerRay = 512;
    constexpr std::size_t kStockMin = 500000;
    constexpr std::size_t kStockMax = 4000000;
    constexpr std::size_t kCutMin = 1000000;
    constexpr std::size_t kCutMax = 6000000;
    constexpr std::size_t kFallbackStock = 2000000;
    constexpr std::size_t kFallbackCut = 4000000;

    std::uint64_t heapBytes = 0;
    if (device)
    {
        if (vsg::PhysicalDevice* pd = device->getPhysicalDevice())
        {
            VkPhysicalDeviceMemoryProperties memProps{};
            vkGetPhysicalDeviceMemoryProperties(pd->vk(), &memProps);
            for (std::uint32_t i = 0; i < memProps.memoryHeapCount; ++i)
            {
                if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    heapBytes += memProps.memoryHeaps[i].size;
            }
        }
    }

    const unsigned cores = std::thread::hardware_concurrency();
    std::size_t stock = kFallbackStock;
    std::size_t cut = kFallbackCut;

    if (heapBytes > 0)
    {
        const double raw = (static_cast<double>(heapBytes) * 0.20) / static_cast<double>(kBytesPerRay);
        stock = static_cast<std::size_t>(raw);
        // Round down to nearest 100k for stable logging.
        stock = (stock / 100000) * 100000;
        if (stock < kStockMin) stock = kStockMin;
        if (stock > kStockMax) stock = kStockMax;

        const double cutMul = (cores > 0 && cores < 4) ? 1.5 : 2.0;
        cut = static_cast<std::size_t>(static_cast<double>(stock) * cutMul);
        if (cut < kCutMin) cut = kCutMin;
        if (cut > kCutMax) cut = kCutMax;
    }

    _maxRenderedRays = stock;
    _maxCutFaceRays = cut;

    const double heapGiB = static_cast<double>(heapBytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Ray-GS budgets: rendered=" << _maxRenderedRays
              << " cutFace=" << _maxCutFaceRays
              << " (heap=" << heapGiB << " GiB, cores=" << cores << ")\n";
    std::cout.flush();
}

void RenderManager::updateWorldAxesSpecFromStock()
{
    // Length = half the fitted stock AABB diagonal; radii scale with length so
    // the gizmo stays readable without dominating the stock.
    const BoundingBox world = worldStockAabb();
    const double diag = world.valid() ? world.diagonal() : 0.0;
    if (!(diag > 0.0))
    {
        _axesSpec = WorldAxesSpec{};
        return;
    }

    const float length = static_cast<float>(diag * 0.5);
    _axesSpec.length = length;
    _axesSpec.tubeRadius = length * 0.018f;
    _axesSpec.coneRadius = length * 0.040f;
    _axesSpec.coneLength = length * 0.18f;
}

void RenderManager::showWorldAxes(bool show)
{
    if (_axesNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _axesNode), children.end());
        _axesNode = nullptr;
    }
    if (!show)
    {
        if (_viewer) _viewer->request();
        return;
    }

    updateWorldAxesSpecFromStock();

    // Dedicated scene subgraph, separate from stock / tool / trajectory.
    // One child group per axis so X / Y / Z stay independently addressable.
    _axesNode = vsg::Group::create();
    const auto meshes = buildWorldAxesMeshes(_axesSpec, _axesSelected);
    const int selectedAxis = worldAxisIndex(_axesSelected);
    for (int axis = 0; axis < 3; ++axis)
    {
        auto axisNode = vsg::Group::create();
        const bool selected = selectedAxis == axis;
        for (int part = 0; part < 2; ++part)
        {
            const int i = axis * 2 + part;
            if (auto node = toolMeshNode(meshes[static_cast<std::size_t>(i)],
                                         worldAxisColor(axis, selected)))
                axisNode->addChild(node);
        }
        _axesNode->addChild(axisNode);
    }

    if (_viewer && _viewer->compileManager)
    {
        auto compiled = _viewer->compileManager->compile(_axesNode);
        if (compiled) vsg::updateViewer(*_viewer, compiled);
    }
    _scene->addChild(_axesNode);
    if (_viewer) _viewer->request();
}

std::optional<WorldAxisPart> RenderManager::pickWorldAxis(const vsg::Camera& camera, int32_t x,
                                                          int32_t y) const
{
    if (!_axesNode) return std::nullopt;

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
    return pickWorldAxes(_axesSpec, worldNear, worldFar - worldNear);
}

void RenderManager::selectWorldAxis(WorldAxisPart part)
{
    if (_axesSelected == part) return;
    _axesSelected = part;
    if (_axesNode) showWorldAxes(true);
}

vsg::ref_ptr<vsg::Node> RenderManager::buildDrawable(vsg::ref_ptr<vsg::vec3Array> positions,
                                                     vsg::ref_ptr<vsg::vec3Array> normals,
                                                     vsg::ref_ptr<vsg::vec4Array> colors,
                                                     VkVertexInputRate colorRate,
                                                     vsg::ref_ptr<vsg::uintArray> indices,
                                                     bool lines,
                                                     bool transparent,
                                                     bool overlay,
                                                     vsg::ref_ptr<vsg::VertexIndexDraw>* outDraw) const
{
    auto texcoords = vsg::vec2Array::create(positions->size(), vsg::vec2(0.0f, 0.0f));

    // Graphics pipeline via the standard phong shader set.
    auto shaderSet = vsg::createPhongShaderSet(_options);
    auto gpc = vsg::GraphicsPipelineConfigurator::create(shaderSet);

    if (const auto& materialBinding = shaderSet->getDescriptorBinding("material"))
    {
        vsg::ref_ptr<vsg::Data> mat = materialBinding.data;
        if (!mat) mat = vsg::PhongMaterialValue::create();
        if (overlay)
        {
            auto material = vsg::PhongMaterialValue::create();
            auto& phong = material->value();
            phong.ambient = _toolColor;
            phong.diffuse = _toolColor;
            phong.emissive = _toolColor;
            phong.specular = vsg::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            mat = material;
        }
        gpc->assignDescriptor("material", mat);
    }

    gpc->enableArray("vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, 8);
    gpc->enableArray("vsg_Color", colorRate, 16);

    if (lines || transparent || overlay)
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
            bool overlay = false;

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
                if (overlay)
                {
                    dss.depthTestEnable = VK_FALSE;
                    dss.depthWriteEnable = VK_FALSE;
                }
            }
        } setDrawStates;
        setDrawStates.lines = lines;
        setDrawStates.transparent = transparent;
        setDrawStates.overlay = overlay;

        gpc->accept(setDrawStates);
    }

    gpc->init();

    auto vid = vsg::VertexIndexDraw::create();
    vsg::DataList arrays{positions, normals, texcoords, colors};
    vid->assignArrays(arrays);
    vid->assignIndices(indices);
    vid->indexCount = static_cast<uint32_t>(indices->size());
    vid->instanceCount = 1;
    if (outDraw) *outDraw = vid;

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

vsg::dmat4 RenderManager::currentFitMatrix() const
{
    BoundingBox model;
    if (_rayModel && _rayModel->bounds().valid())
        model = _rayModel->bounds();
    else if (_current)
        model = BoundingBox::fromBRep(*_current);
    return fitMatrix(model);
}

BoundingBox RenderManager::worldStockAabb() const
{
    BoundingBox model;
    if (_rayModel && _rayModel->bounds().valid())
        model = _rayModel->bounds();
    else if (_current)
        model = BoundingBox::fromBRep(*_current);
    if (!model.valid()) return {};

    const vsg::dmat4 toWorld = fitMatrix(model);
    BoundingBox world;
    const Point3d& mn = model.min();
    const Point3d& mx = model.max();
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                const vsg::dvec3 corner =
                    toWorld * vsg::dvec3(ix ? mx[0] : mn[0], iy ? mx[1] : mn[1],
                                         iz ? mx[2] : mn[2]);
                world.expand(Point3d{corner.x, corner.y, corner.z});
            }
    return world;
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

    const int stride = rayModel.strideForRayBudget(maxRenderedRays());
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
                    (*colors)[point] = iv.cutBegin() ? _toolColor : axisColor;
                    ++point;
                    (*positions)[point] = vsg::vec3(static_cast<float>(end[0]),
                                                    static_cast<float>(end[1]),
                                                    static_cast<float>(end[2]));
                    (*colors)[point] = iv.cutEnd() ? _toolColor : axisColor;
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
    const float strideF = static_cast<float>(stride < 1 ? 1 : stride);
    if (_viewMode == ViewMode::Disk)
    {
        // Hard disks seal with the circular footprint — sized above half-cellDiag
        // so grazing angles and thinned-out rim endpoints still cover stride gaps.
        const Point3d& resolution = rayModel.resolution();
        double fit = 1.0;
        if (_fitToUnitBox && rayModel.bounds().valid())
        {
            const BoundingBox& bounds = rayModel.bounds();
            const double maxExtent =
                std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
            if (maxExtent > 0.0) fit = maxExtent;
        }
        auto diskRadius = [&](std::size_t axis) -> float {
            const double du = resolution[(axis + 1) % 3];
            const double dv = resolution[(axis + 2) % 3];
            const double cellDiag = std::sqrt(du * du + dv * dv);
            // 1.75× half-diag covers grazing views / edge-clipped discs without
            // restoring the full Gaussian ×2×1.4 footprint that overdrew.
            return static_cast<float>(0.5 * cellDiag * static_cast<double>(strideF) * 1.75 / fit);
        };
        return {diskRadius(0), diskRadius(1), diskRadius(2)};
    }

    // Extra 1.4× so soft Gaussian rims still seal after stride sampling.
    const float s = strideF * 1.4f;
    return {splatRadius(rayModel, 0) * s, splatRadius(rayModel, 1) * s,
            splatRadius(rayModel, 2) * s};
}

SplatStyle RenderManager::splatStyle() const
{
    SplatStyle style;
    style.stockColor = _splatColor;
    style.toolColor = _toolColor;
    // Hard disks should read as a solid metal sheet; Gaussian keeps soft opacity.
    style.opacity = (_viewMode == ViewMode::Disk) ? 1.0f : _splatOpacity;
    return style;
}

void RenderManager::rebuildSplatCache()
{
    if (!_rayModel || _rayModel->rayCount() == 0) return;

    _splatCache.setPointRenderMode(_viewMode == ViewMode::Disk ? PointRenderMode::HardDiskWithAA
                                                               : PointRenderMode::Gaussian);

    const BooleanOp op = Parameter::instance().booleanOp();
    const SplatViewCull viewCull = splatViewCull();
    const int stride = displayStride();
    const int cutStride = cutFaceStride();
    const bool skipCutSplats = skipCutSplatEnds(_splatCache.hasCutFace());
    _splatCache.rebuild(*_rayModel, stride, splatRadii(*_rayModel, stride), splatStyle(),
                        skipCutSplats, viewCull);
    if (op == BooleanOp::Inspection)
        syncInspectionSectionGrid(_inspectionPrevAabb);
    else if (cutMeshEnabled())
    {
        if (!_splatCache.hasCutFace())
            _splatCache.rebuildCutFace(*_rayModel, cutStride, splatStyle().toolColor);
        else
            _splatCache.showCutFace();
    }
    else if (_splatCache.hasCutFace())
    {
        // Display off: drop overlay so cut-tagged disks show through.
        _splatCache.clearCutFace();
        _splatCache.rebuild(*_rayModel, stride, splatRadii(*_rayModel, stride), splatStyle(),
                            /*skipCutSplats=*/false, viewCull);
    }
    presentSplatCache();
}

void RenderManager::refreshCutMeshDisplay()
{
    if (!usesSplatView(_viewMode)) return;
    rebuildSplatCache();
}

void RenderManager::refreshSplatViewForCamera()
{
    // View cull is off: do not rebuild or free discs when the camera moves.
}

vsg::dmat4 RenderManager::modelToClipMatrix() const
{
    if (!_camera || !_camera->projectionMatrix || !_camera->viewMatrix || !_rayModel)
        return {};

    const BoundingBox stock = _rayModel->bounds();
    if (!stock.valid()) return {};

    const vsg::dmat4 modelToWorld = fitMatrix(stock);
    const vsg::dmat4 projectionMatrix = _camera->projectionMatrix->transform();
    const vsg::dmat4 viewMatrix = _camera->viewMatrix->transform();
    return projectionMatrix * viewMatrix * modelToWorld;
}

int RenderManager::coarseStride() const
{
    if (!_rayModel) return 1;
    return _rayModel->strideForRayBudget(maxRenderedRays());
}

int RenderManager::cutFaceStride() const
{
    if (!_rayModel) return 1;
    return _rayModel->strideForRayBudget(maxCutFaceRays());
}

SplatViewCull RenderManager::splatViewCull() const
{
    // Off: NDC frustum cull on zoom/rebuild was dropping untagged stock.
    return {};
}

std::size_t RenderManager::rayCountVisibleAtStride(int stride) const
{
    if (!_rayModel) return 0;
    if (stride < 1) stride = 1;

    const SplatViewCull cull = splatViewCull();
    if (!cull.enabled) return _rayModel->rayCountAtStride(stride);

    auto lock = _rayModel->lockChains();
    const BoundingBox stock = _rayModel->bounds();
    if (!stock.valid()) return 0;

    std::size_t total = 0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = _rayModel->grid(axis);
        if (!grid || grid->empty()) continue;

        const std::size_t u = (axis + 1) % 3;
        const std::size_t v = (axis + 2) % 3;
        Point3d a{0.0, 0.0, 0.0};
        Point3d b{0.0, 0.0, 0.0};
        a[axis] = stock.min()[axis];
        b[axis] = stock.max()[axis];

        for (std::uint32_t iv = 0; iv < grid->height; ++iv)
        {
            if (static_cast<int>(iv) % stride != 0) continue;
            for (std::uint32_t iu = 0; iu < grid->width; ++iu)
            {
                if (static_cast<int>(iu) % stride != 0) continue;
                const RaySlot& slot = grid->at(iu, iv);
                if (slot.empty()) continue;

                a[u] = b[u] = grid->sampleU(iu);
                a[v] = b[v] = grid->sampleV(iv);
                const double da =
                    (a[0] - cull.eyeModel.x) * (a[0] - cull.eyeModel.x) +
                    (a[1] - cull.eyeModel.y) * (a[1] - cull.eyeModel.y) +
                    (a[2] - cull.eyeModel.z) * (a[2] - cull.eyeModel.z);
                const double db =
                    (b[0] - cull.eyeModel.x) * (b[0] - cull.eyeModel.x) +
                    (b[1] - cull.eyeModel.y) * (b[1] - cull.eyeModel.y) +
                    (b[2] - cull.eyeModel.z) * (b[2] - cull.eyeModel.z);
                const Point3d& nearPt = da <= db ? a : b;

                const vsg::dvec4 clip =
                    cull.modelToClip * vsg::dvec4(nearPt[0], nearPt[1], nearPt[2], 1.0);
                if (clip.w <= 1.0e-12) continue;
                const double invW = 1.0 / clip.w;
                const double ndcX = clip.x * invW;
                const double ndcY = clip.y * invW;
                const double m = cull.ndcMargin;
                if (ndcX < -1.0 - m || ndcX > 1.0 + m || ndcY < -1.0 - m || ndcY > 1.0 + m)
                    continue;

                // Budget the near-face densify only; far ends stay on coarseStride.
                total += slot.intervalCount;
            }
        }
    }
    return total;
}

BoundingBox RenderManager::visibleStockAabb() const
{
    if (!_rayModel || !_rayModel->bounds().valid()) return {};

    const BoundingBox stock = _rayModel->bounds();
    if (!_camera || !_camera->projectionMatrix || !_camera->viewMatrix) return stock;

    const vsg::dmat4 modelToWorld = fitMatrix(stock);
    const vsg::dmat4 worldToModel = vsg::inverse(modelToWorld);
    const vsg::dmat4 projectionMatrix = _camera->projectionMatrix->transform();
    const vsg::dmat4 viewMatrix = _camera->viewMatrix->transform();
    const auto viewport = _camera->getViewport();
    const bool reverseDepth = projectionMatrix(2, 2) > 0.0;
    const double zNear = reverseDepth ? viewport.maxDepth : viewport.minDepth;
    const double zFar = reverseDepth ? viewport.minDepth : viewport.maxDepth;

    const vsg::dmat4 invViewProj = vsg::inverse(projectionMatrix * viewMatrix);
    BoundingBox frustumModel;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                const vsg::dvec4 clip =
                    invViewProj * vsg::dvec4(ix ? 1.0 : -1.0, iy ? 1.0 : -1.0,
                                            iz ? zFar : zNear, 1.0);
                if (std::abs(clip.w) < 1.0e-12) continue;
                const vsg::dvec3 world(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
                const vsg::dvec3 model = worldToModel * world;
                frustumModel.expand(Point3d{model.x, model.y, model.z});
            }

    BoundingBox visible = intersectAabb(stock, frustumModel);
    if (!visible.valid()) return {};

    const Point3d& res = _rayModel->resolution();
    const double pad = 2.0 * std::max({res[0], res[1], res[2], 1.0e-9});
    return expandAabb(visible, pad);
}

int RenderManager::displayStride() const
{
    if (!_rayModel) return 1;
    if (!_camera) return _rayModel->strideForRayBudget(maxRenderedRays());

    if (maxRenderedRays() == 0) return 1;
    const std::size_t total = rayCountVisibleAtStride(1);
    if (total <= maxRenderedRays()) return 1;

    const double estimate = std::sqrt(static_cast<double>(total) / static_cast<double>(maxRenderedRays()));
    int stride = (estimate > 1.0) ? static_cast<int>(estimate) : 1;
    while (stride < RayModel::maxStride && rayCountVisibleAtStride(stride) > maxRenderedRays())
        ++stride;
    return stride;
}

void RenderManager::setCamera(vsg::ref_ptr<vsg::Camera> camera)
{
    _camera = std::move(camera);
    if (!_splatViewDebounce)
    {
        _splatViewDebounce = new QTimer();
        _splatViewDebounce->setSingleShot(true);
        _splatViewDebounce->setInterval(splatViewDebounceMs);
        QObject::connect(_splatViewDebounce, &QTimer::timeout, [this]() {
            refreshSplatViewForCamera();
        });
    }
}

void RenderManager::noteCameraMoved()
{
    if (!usesSplatView(_viewMode) || !_camera) return;
    if (!_splatViewDebounce) setCamera(_camera);
    if (_splatViewDebounce) _splatViewDebounce->start();
}

namespace
{

struct CameraSettleHandler : public vsg::Inherit<vsg::Visitor, CameraSettleHandler>
{
    explicit CameraSettleHandler(RenderManager* manager) : _manager(manager) {}

    void apply(vsg::ButtonPressEvent& /*event*/) override { kick(); }
    void apply(vsg::ButtonReleaseEvent& /*event*/) override { kick(); }
    void apply(vsg::MoveEvent& event) override
    {
        if (event.mask != 0) kick();
    }
    void apply(vsg::ScrollWheelEvent& /*event*/) override { kick(); }

    void kick()
    {
        if (_manager) _manager->noteCameraMoved();
    }

    RenderManager* _manager = nullptr;
};

} // namespace

vsg::ref_ptr<vsg::Visitor> RenderManager::createCameraSettleHandler()
{
    return CameraSettleHandler::create(this);
}

bool RenderManager::splatOnScreen() const
{
    if (!_modelNode || !_splatCache.node()) return false;
    if (_modelNode == _splatCache.node()) return true;
    const auto* xform = dynamic_cast<const vsg::MatrixTransform*>(_modelNode.get());
    if (!xform || xform->children.empty()) return false;
    return xform->children.front() == _splatCache.node();
}

void RenderManager::presentSplatCache()
{
    if (!_splatCache.node() || !_rayModel) return;
    if (_splatCache.gpuNeedsCompile() || !splatOnScreen())
    {
        // Only drop the compile flag once the arrays really are on the device.
        // Clearing it after a failed compile lets the next frame record a draw
        // with unbacked BufferInfos (SIGSEGV at BufferInfo::buffer + 0x30).
        _splatCache.prepareGpuCompile();
        if (attach(applyFit(_splatCache.node(), _rayModel->bounds()), true))
            _splatCache.noteCompiled();
        else
            _splatCache.revertFailedGpuCompile();
        return;
    }
    if (_viewer) _viewer->request();
}

void RenderManager::syncInspectionSectionGrid(const BoundingBox& sectionAabb)
{
    if (!usesSplatView(_viewMode) || !_rayModel ||
        Parameter::instance().booleanOp() != BooleanOp::Inspection || !sectionAabb.valid())
    {
        _splatCache.clearSectionGrid();
        return;
    }

    const int stride = coarseStride();
    _splatCache.updateSectionGrid(*_rayModel, stride, sectionAabb, splatStyle().stockColor);
}

vsg::ref_ptr<vsg::Node> RenderManager::createSplatNode(const RayModel& rayModel) const
{
    auto chainLock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
        throw std::runtime_error("The ray model contains no rays; try a coarser resolution.");

    const int stride = rayModel.strideForRayBudget(maxRenderedRays());

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
                    if (!span.hasSolidLength()) continue;
                    Point3d start{0.0, 0.0, 0.0};
                    Point3d end{0.0, 0.0, 0.0};
                    start[axis] = grid->fromTick(span.begin);
                    end[axis] = grid->fromTick(span.end);
                    start[u] = u0;
                    end[u] = u0;
                    start[v] = v0;
                    end[v] = v0;

                    const double cellDiag = std::sqrt(
                        static_cast<double>(grid->spacingU) * static_cast<double>(grid->spacingU) +
                        static_cast<double>(grid->spacingV) * static_cast<double>(grid->spacingV));
                    const float spanRadius =
                        splatRadiusForSpan(radius, end[axis] - start[axis], cellDiag, stride);

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

                    splats.push_back({vsg::vec3(static_cast<float>(start[0]),
                                                static_cast<float>(start[1]),
                                                static_cast<float>(start[2])),
                                      normalOrAxis(span.beginNormal, true),
                                      span.cutBegin() ? toolColor : color, spanRadius});
                    splats.push_back({vsg::vec3(static_cast<float>(end[0]),
                                                static_cast<float>(end[1]),
                                                static_cast<float>(end[2])),
                                      normalOrAxis(span.endNormal, false),
                                      span.cutEnd() ? toolColor : color, spanRadius});
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
    clearTrajectory();
    rebuild();

    // A new model may change units; the caller reseeds Parameter::toolRadius
    // first, then this rebuilds the mesh if a cutter is active.
    if (_toolType != ToolType::None) rebuildTool(true);

    // Rescale the XYZ gizmo to the new stock AABB.
    if (_axesNode) showWorldAxes(true);
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
    _preShellRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();

    const auto replaced = _rayModels.find(resolution);
    if (replaced != _rayModels.end()) _cachedRays -= replaced->second.rayCount();

    _rayModels.insert_or_assign(resolution, std::move(model));
    _cachedRays += rays;

    touchRayModel(resolution);
    _sourceRayModel = &_rayModels.at(resolution);
    if (_profiling)
    {
        const PairingStats& s = _sourceRayModel->pairingStats();
        std::printf("cast pairing  odd-rays %lld  unmatched enter %lld leave %lld  slivers %lld\n",
                    s.oddHitRays, s.unmatchedEnter, s.unmatchedLeave, s.sliversDropped);
        std::fflush(stdout);
    }
    applyBooleanToRayModel();
}

bool RenderManager::useCachedRayModel(const Point3d& resolution)
{
    if (_rayModels.find(resolution) == _rayModels.end()) return false;

    _rayModel = nullptr;
    _sourceRayModel = nullptr;
    _booleanRayModel.reset();
    _preShellRayModel.reset();
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
    _preShellRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
    _rayModels.clear();
    _rayModelOrder.clear();
    _cachedRays = 0;
    // Drop the VSG splat subgraph and GPU arrays so the next model cannot
    // inherit compiled slots from this one.
    _splatCache.release();
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
        if (usesSplatView(_viewMode))
        {
            rebuildSplatCache();
            return;
        }

        attach(createRayNode(*_rayModel), true);
        return;
    }

    if (_current) attach(createNode(*_current), true);
}

void RenderManager::refreshRayViewsAfterStockEdit()
{
    if (usesSplatView(_viewMode))
    {
        if (_rayModel && _rayModel->rayCount() > 0)
            rebuildSplatCache();
        else
        {
            _splatCache.clear();
            if (_current) attach(createNode(*_current), true);
            else if (_viewer) _viewer->request();
        }
        return;
    }

    if (_viewMode == ViewMode::Ray)
    {
        // Drop any leftover GS subgraph so line mode shows the edited stock.
        _splatCache.clear();
        if (_rayModel && _rayModel->rayCount() > 0)
            attach(createRayNode(*_rayModel), true);
        else if (_current)
            attach(createNode(*_current), true);
        else if (_viewer)
            _viewer->request();
        return;
    }

    if (_viewer) _viewer->request();
}

void RenderManager::clear()
{
    _scene->children.clear();
    _modelNode = nullptr;
    _toolTransform = nullptr;
    _sweptNode = nullptr;
    _axesNode = nullptr;
    _sweptVolume.reset();
    _cutSweep.reset();
    _lastToolPose.reset();
    _lastWheelY.reset();
    clearTrajectory();
    _current.reset();
    clearRayModels();
    showWorldAxes(true);
    if (_viewer) _viewer->request();
}

void RenderManager::setToolType(ToolType type)
{
    if (type == _toolType) return;
    _toolType = type;
    if (type == ToolType::None)
    {
        _lastToolPose.reset();
        _lastWheelY.reset();
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

void RenderManager::setToolColor(const vsg::vec4& color)
{
    if (_toolColor == color) return;
    _toolColor = color;
    if (_toolType != ToolType::None)
        rebuildTool(true);
}

float RenderManager::worldFromModelLength(double value) const
{
    if (!(value > 0.0)) return 0.0f;

    // Parameter stores model-space sizes; the tool mesh is placed in world
    // space beside the fitted model, so apply the same scale as applyFit().
    if (_fitToUnitBox && _current)
    {
        const BoundingBox bounds = BoundingBox::fromBRep(*_current);
        if (bounds.valid())
        {
            const double maxExtent =
                std::max({bounds.extent(0), bounds.extent(1), bounds.extent(2)});
            if (maxExtent > 0.0) value /= maxExtent;
        }
    }

    return static_cast<float>(value);
}

float RenderManager::worldToolRadius() const
{
    double radius = Parameter::instance().toolRadius();
    if (radius <= 0.0) radius = 0.05;
    return worldFromModelLength(radius);
}

float RenderManager::worldToolLength() const
{
    double length = Parameter::instance().toolLength();
    if (length <= 0.0) length = static_cast<double>(worldToolRadius()) * 2.8;
    return worldFromModelLength(length);
}

GrindingWheelProfile RenderManager::worldGrindingWheelProfile() const
{
    const auto& store = Parameter::instance();
    GrindingWheelProfile wheel;
    wheel.tipWidth = worldFromModelLength(store.toolWheelTipWidth());
    wheel.shoulderWidth = worldFromModelLength(store.toolWheelShoulderWidth());
    wheel.taperHeight = worldFromModelLength(store.toolWheelTaperHeight());
    wheel.shoulderHeight = worldFromModelLength(store.toolWheelShoulderHeight());
    return wheel;
}

vsg::ref_ptr<vsg::Node> RenderManager::toolMeshNode(const TriangleMesh& mesh,
                                                    const vsg::vec4& color) const
{
    if (mesh.triangles.empty()) return {};

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

    auto colors = vsg::vec4Array::create(1, color);
    auto indices = vsg::uintArray::create(faceVertices.size());
    for (std::size_t i = 0; i < faceVertices.size(); ++i) (*indices)[i] = faceVertices[i];

    return buildDrawable(positions, normals, colors, VK_VERTEX_INPUT_RATE_INSTANCE,
                         indices, false);
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
    const float vertexAngle =
        static_cast<float>(Parameter::instance().toolVertexAngleDeg());
    const float shankRadius =
        worldFromModelLength(Parameter::instance().toolShankRadius());
    const float shankLength =
        worldFromModelLength(Parameter::instance().toolShankLength());
    const TriangleMesh mesh = createToolMesh(_toolType, radius, height, 48, 24, 12,
                                             vertexAngle, shankRadius, shankLength,
                                             worldGrindingWheelProfile());
    auto drawable = toolMeshNode(mesh, _toolColor);
    if (!drawable) return;

    _toolTransform = vsg::MatrixTransform::create();
    _toolTransform->addChild(drawable);

    if (shankRadius > 0.0f && shankLength > 0.0f)
    {
        float drawnShankRadius = shankRadius;
        if (_toolType == ToolType::Sphere && drawnShankRadius >= radius)
            drawnShankRadius = radius * 0.6f;
        TriangleMesh shankMesh;
        if (_toolType == ToolType::GrindingWheel)
            shankMesh = createGrindingShankMesh(worldGrindingWheelProfile().shoulderHeight,
                                               drawnShankRadius, shankLength);
        else
        {
            const float z0 = (_toolType == ToolType::Sphere)
                                 ? radius
                                 : toolCuttingTop(_toolType, radius, height, vertexAngle);
            shankMesh = createShankMesh(drawnShankRadius, z0, shankLength);
        }
        if (auto shankNode = toolMeshNode(shankMesh, _shankColor))
            _toolTransform->addChild(shankNode);
    }

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

void RenderManager::setToolPose(const vsg::dvec3& position, const vsg::dvec3& direction,
                                const vsg::dvec3* alongHint)
{
    vsg::dvec3 tip, x, y, z;
    if (!referencePoseToTipFrame(position, direction, tip, x, y, z)) return;
    if (_toolType == ToolType::GrindingWheel)
    {
        vsg::dvec3 along(0.0, 0.0, 0.0);
        if (alongHint && vsg::length(*alongHint) > 1.0e-12)
            along = *alongHint;
        else if (_lastToolPose)
            along = tip - _lastToolPose->position;
        const vsg::dvec3* prevY = _lastWheelY ? &*_lastWheelY : nullptr;
        if (vsg::length(along) > 1.0e-12 || prevY)
            grindingWheelMotionFrame(z, along, x, y, z, prevY);
    }
    commitToolTip(tip, x, y, z);
}

bool RenderManager::referencePoseToTipFrame(const vsg::dvec3& position, const vsg::dvec3& direction,
                                            vsg::dvec3& tip, vsg::dvec3& x, vsg::dvec3& y,
                                            vsg::dvec3& z) const
{
    if (!_toolTransform || _toolType == ToolType::None) return false;

    z = direction;
    const double zLen = vsg::length(z);
    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 0.0);
    else z /= zLen;

    if (_toolType == ToolType::GrindingWheel)
    {
        // CL is the shoulder-rectangle midpoint. Table `direction` is the
        // spindle; local +Z is radial from that axis through the CL so the
        // outer rim points toward the stock. Feed yaw is applied afterwards.
        if (zLen <= 0.0) z = vsg::dvec3(1.0, 0.0, 0.0);
        x = z; // spindle = local +X
        BoundingBox world = worldStockAabb();
        vsg::dvec3 origin(0.0, 0.0, 0.0);
        if (world.valid())
        {
            const Point3d c = world.centre();
            origin = vsg::dvec3(c[0], c[1], c[2]);
        }
        vsg::dvec3 rel = position - origin;
        rel = rel - x * vsg::dot(rel, x);
        const double relLen = vsg::length(rel);
        if (relLen > 1.0e-9)
            z = rel / relLen; // outward radial → local +Z
        else
        {
            vsg::dvec3 up(0.0, 0.0, 1.0);
            if (std::abs(vsg::dot(x, up)) > 0.95) up = vsg::dvec3(0.0, 1.0, 0.0);
            z = vsg::normalize(vsg::cross(up, x));
        }
        y = vsg::cross(z, x);
        const double yLen = vsg::length(y);
        if (yLen > 0.0) y /= yLen;
        else y = vsg::dvec3(0.0, 1.0, 0.0);
        z = vsg::cross(x, y); // re-orthonormalize
        tip = position;
        return true;
    }

    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);

    // Prefer world +Z as the reference "up" when building the tool frame; fall
    // back to +X when the axis is nearly vertical.
    vsg::dvec3 up(0.0, 0.0, 1.0);
    if (std::abs(vsg::dot(z, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);

    x = vsg::cross(up, z);
    const double xLen = vsg::length(x);
    if (xLen <= 0.0) x = vsg::dvec3(1.0, 0.0, 0.0);
    else x /= xLen;

    y = vsg::cross(z, x);

    // Mesh / sweep frames put the tip at the origin. The mouse / table hit is
    // the sphere or fillet centre for ball, sphere, and bull, so shift the tip
    // down the axis by that offset.
    tip = position;
    const double centerOffset = static_cast<double>(toolCenterOffset(_toolType, worldToolRadius()));
    if (centerOffset > 0.0)
        tip = position - z * centerOffset;
    return true;
}

void RenderManager::setToolPosePath(const std::vector<ToolPose>& referencePoses)
{
    if (!_toolTransform || _toolType == ToolType::None) return;
    if (referencePoses.empty()) return;
    if (referencePoses.size() == 1)
    {
        setToolPose(referencePoses.front().position, referencePoses.front().direction);
        return;
    }

    std::vector<ToolPose> tipPoses;
    tipPoses.reserve(referencePoses.size());
    vsg::dvec3 tip, x, y, z;
    for (const ToolPose& ref : referencePoses)
    {
        if (!referencePoseToTipFrame(ref.position, ref.direction, tip, x, y, z)) return;
        // Grinding sweep stores radial as direction (local +Z); others store axis.
        ToolPose tipPose{tip, z};
        tipPose.feed = ref.feed;
        tipPoses.push_back(tipPose);
    }

    if (_toolType == ToolType::GrindingWheel && tipPoses.size() >= 2)
    {
        const ToolPose& last = tipPoses.back();
        const ToolPose& prev = tipPoses[tipPoses.size() - 2];
        const vsg::dvec3 along = (vsg::length(last.feed) > 1.0e-12)
                                     ? last.feed
                                     : (last.position - prev.position);
        const vsg::dvec3* prevY = _lastWheelY ? &*_lastWheelY : nullptr;
        grindingWheelMotionFrame(z, along, x, y, z, prevY);
    }

    _toolTransform->matrix = vsg::dmat4(x.x, x.y, x.z, 0.0,
                                        y.x, y.y, y.z, 0.0,
                                        z.x, z.y, z.z, 0.0,
                                        tip.x, tip.y, tip.z, 1.0);
    _lastToolPose = tipPoses.back();
    if (_toolType == ToolType::GrindingWheel)
        _lastWheelY = y;
    else
        _lastWheelY.reset();
    for (const ToolPose& p : tipPoses)
        appendToolTrajectory(p.position);

    if (Parameter::instance().booleanOp() == BooleanOp::Inspection)
    {
        const bool movedEnough = placeInspectionCutter(tipPoses.back(), false);
        if (movedEnough)
            applyBooleanToRayModel();
        else if (_viewer)
            _viewer->request();
        return;
    }

    const bool sweepChanged = recordSweepPath(tipPoses);
    if (sweepChanged)
        applyBooleanToRayModel();
    else if (_viewer)
        _viewer->request();
}

void RenderManager::setToolTipPose(const ToolPose& pose)
{
    if (!_toolTransform || _toolType == ToolType::None) return;

    vsg::dvec3 z = pose.direction;
    const double zLen = vsg::length(z);
    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);
    else z /= zLen;

    vsg::dvec3 up(0.0, 0.0, 1.0);
    if (std::abs(vsg::dot(z, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);

    vsg::dvec3 x = vsg::cross(up, z);
    const double xLen = vsg::length(x);
    if (xLen <= 0.0) x = vsg::dvec3(1.0, 0.0, 0.0);
    else x /= xLen;

    commitToolTip(pose.position, x, vsg::cross(z, x), z);
}

std::optional<ToolPose> RenderManager::lastReferencePose() const
{
    if (!_lastToolPose) return std::nullopt;

    ToolPose pose = *_lastToolPose;
    const double offset = static_cast<double>(toolCenterOffset(_toolType, worldToolRadius()));
    if (offset <= 0.0) return pose;

    vsg::dvec3 z = pose.direction;
    const double zLen = vsg::length(z);
    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);
    else z /= zLen;
    pose.position = pose.position + z * offset;
    return pose;
}

void RenderManager::resetSweepAnchor()
{
    if (_sweptNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _sweptNode), children.end());
        _sweptNode = nullptr;
    }
    _cutSweep.reset();
    if (_toolType != ToolType::None)
        _sweptVolume = SweptVolume{};
    else
        _sweptVolume.reset();
}

void RenderManager::resetBooleanStock()
{
    _booleanRayModel.reset();
    _preShellRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
    _rayModel = _sourceRayModel;
    _splatCache.clearCutFace();

    if (!usesRayModel(_viewMode) || !_rayModel || _rayModel->rayCount() == 0)
    {
        if (_viewer) _viewer->request();
        return;
    }

    // Packed rebuild of the source cast. Do not updateRegion the stock AABB:
    // that keeps _allocEnd at the cut high-water mark and submits holes.
    // Do not rebuildCutFace: the source has no cut overlay.
    if (usesSplatView(_viewMode))
    {
        const int stride = displayStride();
        _splatCache.setPointRenderMode(_viewMode == ViewMode::Disk
                                           ? PointRenderMode::HardDiskWithAA
                                           : PointRenderMode::Gaussian);
        _splatCache.rebuild(*_rayModel, stride, splatRadii(*_rayModel, stride), splatStyle(),
                            skipCutSplatEnds(false), splatViewCull());
        presentSplatCache();
        return;
    }

    rebuild();
}

bool RenderManager::shellStock(double thickness)
{
    if (!(thickness > 0.0)) return false;

    // Re-shelling (new thickness): restore the pre-shell stock first so walls
    // are not stacked on an already-hollowed model. First Shell: cache current.
    if (_preShellRayModel)
    {
        _booleanRayModel = _preShellRayModel->clone();
    }
    else
    {
        const RayModel* current = nullptr;
        if (_booleanRayModel) current = &*_booleanRayModel;
        else if (_sourceRayModel) current = _sourceRayModel;
        if (!current || current->rayCount() == 0) return false;

        _preShellRayModel = current->clone();
        if (!_booleanRayModel)
            _booleanRayModel = _preShellRayModel->clone();
    }

    try
    {
        _booleanRayModel->shellInPlace(thickness);
    }
    catch (const std::exception&)
    {
        if (_preShellRayModel)
            _booleanRayModel = _preShellRayModel->clone();
        return false;
    }

    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
    _rayModel = &*_booleanRayModel;
    _splatCache.clearCutFace();
    refreshRayViewsAfterStockEdit();
    return true;
}

bool RenderManager::cancelShell()
{
    if (!_preShellRayModel) return false;

    _booleanRayModel = std::move(*_preShellRayModel);
    _preShellRayModel.reset();
    _inspectionRayModel.reset();
    _inspectionPrevAabb = {};
    _inspectionBooleanPose.reset();
    _rayModel = &*_booleanRayModel;
    _splatCache.clearCutFace();
    refreshRayViewsAfterStockEdit();
    return true;
}

void RenderManager::retractToolAndResetSweep()
{
    if (_toolTransform && _toolType != ToolType::None && _lastToolPose)
    {
        vsg::dvec3 z = _lastToolPose->direction;
        const double zLen = vsg::length(z);
        if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);
        else z /= zLen;

        const vsg::dvec3 tip = _lastToolPose->position;
        const double radius = static_cast<double>(worldToolRadius());
        const BoundingBox stock = worldStockAabb();
        double t = 0.0;
        if (stock.valid())
        {
            const BoundingBox expanded = expandAabb(stock, radius);
            const Point3d tipPt{tip.x, tip.y, tip.z};
            if (expanded.contains(tipPt))
                t = rayAabbExitT(tip, z, expanded);
            const double extra = std::max(radius, stock.diagonal() * 0.05);
            t += extra;
        }
        else
        {
            t = std::max(4.0 * radius, 1.0);
        }

        const vsg::dvec3 newTip = tip + z * t;

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
                                            newTip.x, newTip.y, newTip.z, 1.0);
        _lastToolPose = ToolPose{newTip, z};
    }

    resetSweepAnchor();
    _trajectoryConnect = false;
    if (_viewer) _viewer->request();
}

void RenderManager::commitToolTip(const vsg::dvec3& tip, const vsg::dvec3& x, const vsg::dvec3& y,
                                  const vsg::dvec3& z)
{
    if (!_toolTransform || _toolType == ToolType::None) return;

    _toolTransform->matrix = vsg::dmat4(x.x, x.y, x.z, 0.0,
                                        y.x, y.y, y.z, 0.0,
                                        z.x, z.y, z.z, 0.0,
                                        tip.x, tip.y, tip.z, 1.0);

    const ToolPose pose{tip, z};
    _lastToolPose = pose;
    if (_toolType == ToolType::GrindingWheel)
        _lastWheelY = y;
    else
        _lastWheelY.reset();
    appendToolTrajectory(tip);

    if (Parameter::instance().booleanOp() == BooleanOp::Inspection)
    {
        const bool movedEnough = placeInspectionCutter(pose, false);
        if (movedEnough)
            applyBooleanToRayModel();
        else if (_viewer)
            _viewer->request();
        return;
    }

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
        (appliesBoolean(Parameter::instance().booleanOp())) ? 0.05 : 0.25;
    if (move < static_cast<double>(radius) * minMoveFactor) return false;

    return recordSweepPath({*sweep.lastPose(), pose});
}

bool RenderManager::recordSweepPath(const std::vector<ToolPose>& tipPoses)
{
    if (_toolType == ToolType::None || tipPoses.size() < 2) return false;

    if (!_sweptVolume) _sweptVolume = SweptVolume{};

    SweptVolume& sweep = *_sweptVolume;

    // Seed-only: remember the first station without cutting when nothing was
    // anchored yet and the caller only handed a path that starts here.
    const bool continuing = static_cast<bool>(sweep.lastPose());
    if (!continuing)
        sweep.setLastPose(tipPoses.front());

    const float radius = worldToolRadius();

    // Consecutive booleans share a pose. Offset the first station back along
    // the incoming feed so this closed solid overlaps the previous leading cap
    // instead of sitting on it. Caps stay; the operand stays watertight.
    std::vector<ToolPose> path = tipPoses;
    if (continuing && path.size() >= 2)
    {
        const vsg::dvec3 delta = path[1].position - path[0].position;
        const double segLen = vsg::length(delta);
        if (segLen > 1.0e-12)
        {
            const RayModel* model = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
            const double eps = backExtendEpsilon(model, static_cast<double>(radius), segLen);
            path[0].position -= (delta / segLen) * eps;
        }
    }

    // Boolean always uses this one path solid. Stock (_booleanRayModel) already
    // holds prior cuts; re-walking the whole history would only grow cost.
    SweptVolume step;
    step.appendPath(_toolType, radius, worldToolLength(), path, SweptVolume::kCircleSegments,
                    static_cast<float>(Parameter::instance().toolVertexAngleDeg()),
                    worldFromModelLength(Parameter::instance().toolShankRadius()),
                    worldFromModelLength(Parameter::instance().toolShankLength()),
                    worldGrindingWheelProfile());
    if (step.empty())
    {
        sweep.setLastPose(tipPoses.back());
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
        sweep.setLastPose(tipPoses.back());
    }

    publishSweptVolume();
    return true;
}

bool RenderManager::placeInspectionCutter(const ToolPose& pose, bool forceBoolean)
{
    if (_toolType == ToolType::None || !_toolTransform) return false;

    TriangleMesh mesh =
        createToolMesh(_toolType, worldToolRadius(), worldToolLength(), 48, 24, 12,
                       static_cast<float>(Parameter::instance().toolVertexAngleDeg()),
                       worldFromModelLength(Parameter::instance().toolShankRadius()),
                       worldFromModelLength(Parameter::instance().toolShankLength()),
                       worldGrindingWheelProfile());
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

void RenderManager::clearTrajectory()
{
    if (_trajectoryNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _trajectoryNode),
                       children.end());
    }
    _trajectoryNode = nullptr;
    _trajectoryDraw = nullptr;
    _trajectoryPositions = nullptr;
    _trajectoryIndices = nullptr;
    _trajectoryPointCount = 0;
    _trajectoryIndexCount = 0;
    _trajectoryConnect = false;
    if (_viewer) _viewer->request();
}

void RenderManager::setTrajectoryPath(const std::vector<vsg::dvec3>& points)
{
    clearTrajectory();
    if (points.size() < 2) return;

    _trajectoryConnect = false;
    for (const vsg::dvec3& p : points)
    {
        const vsg::vec3 point(static_cast<float>(p.x), static_cast<float>(p.y),
                              static_cast<float>(p.z));
        ensureTrajectoryCapacity();
        if (!_trajectoryPositions || !_trajectoryIndices || !_trajectoryDraw) return;

        (*_trajectoryPositions)[_trajectoryPointCount] = point;
        if (_trajectoryConnect && _trajectoryPointCount > 0)
        {
            (*_trajectoryIndices)[_trajectoryIndexCount] = _trajectoryPointCount - 1;
            (*_trajectoryIndices)[_trajectoryIndexCount + 1] = _trajectoryPointCount;
            _trajectoryIndexCount += 2;
            _trajectoryDraw->indexCount = _trajectoryIndexCount;
            _trajectoryIndices->dirty();
        }
        ++_trajectoryPointCount;
        _trajectoryConnect = true;
        _trajectoryPositions->dirty();
    }
    if (_viewer) _viewer->request();
}

void RenderManager::ensureTrajectoryCapacity()
{
    const auto neededPoints = _trajectoryPointCount + 1;
    const auto neededIndices = _trajectoryIndexCount + 2;
    const bool pointsFit =
        _trajectoryPositions && neededPoints <= _trajectoryPositions->size();
    const bool indicesFit =
        _trajectoryIndices && neededIndices <= _trajectoryIndices->size();
    if (pointsFit && indicesFit && _trajectoryDraw) return;

    auto newPointCap = _trajectoryPositions ? _trajectoryPositions->size() : 256;
    if (newPointCap == 0) newPointCap = 256;
    while (newPointCap < neededPoints) newPointCap *= 2;
    auto newIndexCap = _trajectoryIndices ? _trajectoryIndices->size() : 512;
    if (newIndexCap == 0) newIndexCap = 512;
    while (newIndexCap < neededIndices) newIndexCap *= 2;

    auto positions = vsg::vec3Array::create(newPointCap, vsg::vec3(0.0f, 0.0f, 0.0f));
    auto normals = vsg::vec3Array::create(newPointCap, vsg::vec3(0.0f, 0.0f, 1.0f));
    auto indices = vsg::uintArray::create(newIndexCap, 0u);
    positions->properties.dataVariance = vsg::DYNAMIC_DATA;
    indices->properties.dataVariance = vsg::DYNAMIC_DATA;

    if (_trajectoryPositions)
    {
        for (auto i = decltype(newPointCap){0}; i < _trajectoryPointCount; ++i)
            (*positions)[i] = (*_trajectoryPositions)[i];
    }
    if (_trajectoryIndices)
    {
        for (auto i = decltype(newIndexCap){0}; i < _trajectoryIndexCount; ++i)
            (*indices)[i] = (*_trajectoryIndices)[i];
    }

    auto colors = vsg::vec4Array::create(1, _toolColor);
    auto node = buildDrawable(positions, normals, colors, VK_VERTEX_INPUT_RATE_INSTANCE,
                              indices, true, false, true, &_trajectoryDraw);
    if (_trajectoryDraw) _trajectoryDraw->indexCount = _trajectoryIndexCount;

    if (_trajectoryNode)
    {
        auto& children = _scene->children;
        children.erase(std::remove(children.begin(), children.end(), _trajectoryNode),
                       children.end());
    }

    _trajectoryNode = node;
    _trajectoryPositions = positions;
    _trajectoryIndices = indices;

    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(_trajectoryNode);
        if (compileResult) vsg::updateViewer(*_viewer, compileResult);
    }
    _scene->addChild(_trajectoryNode);
}

void RenderManager::appendToolTrajectory(const vsg::dvec3& position)
{
    if (!recordsToolPath(Parameter::instance().booleanOp())) return;
    if (_toolType == ToolType::None) return;

    const vsg::vec3 point(static_cast<float>(position.x), static_cast<float>(position.y),
                          static_cast<float>(position.z));

    if (_trajectoryConnect && _trajectoryPointCount > 0 && _trajectoryPositions)
    {
        const vsg::vec3 delta = point - (*_trajectoryPositions)[_trajectoryPointCount - 1];
        if (vsg::length(delta) < 1.0e-4f) return;
    }

    ensureTrajectoryCapacity();
    if (!_trajectoryPositions || !_trajectoryIndices || !_trajectoryDraw) return;

    (*_trajectoryPositions)[_trajectoryPointCount] = point;

    if (_trajectoryConnect && _trajectoryPointCount > 0)
    {
        (*_trajectoryIndices)[_trajectoryIndexCount] = _trajectoryPointCount - 1;
        (*_trajectoryIndices)[_trajectoryIndexCount + 1] = _trajectoryPointCount;
        _trajectoryIndexCount += 2;
        _trajectoryDraw->indexCount = _trajectoryIndexCount;
        _trajectoryIndices->dirty();
    }

    ++_trajectoryPointCount;
    _trajectoryConnect = true;
    _trajectoryPositions->dirty();
    if (_viewer) _viewer->request();
}

void RenderManager::setBooleanOp(BooleanOp op)
{
    const BooleanOp previous = Parameter::instance().booleanOp();
    Parameter::instance().setBooleanOp(op);
    if (!recordsToolPath(op)) _trajectoryConnect = false;

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

    if (op == BooleanOp::None || op == BooleanOp::Probe)
    {
        // Keep whatever RayModel is on screen; only stop applying new cuts.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
        if (usesSplatView(_viewMode) && _rayModel && _rayModel->rayCount() > 0)
            rebuildSplatCache();
        else if (_viewer)
            _viewer->request();
        return;
    }

    applyBooleanToRayModel();
    // No new segment: still drop/restore the overlay and cut-end splat skip
    // for the op we just entered.
    if ((op == BooleanOp::Subtraction || op == BooleanOp::Union) &&
        (!_cutSweep || _cutSweep->empty()) && usesSplatView(_viewMode) && _rayModel &&
        _rayModel->rayCount() > 0)
        rebuildSplatCache();
}

void RenderManager::applyBooleanToRayModel()
{
    if (!_sourceRayModel)
    {
        _booleanRayModel.reset();
        _preShellRayModel.reset();
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
    double cloneMs = 0.0;
    bool raysMutated = false;

    // Tool motion calls this on every sweep step, so the branches below that
    // only re-point _rayModel must not trigger a redraw of unchanged rays.
    const RayModel* const displayedBefore = _rayModel;
    const bool wasInspecting =
        _inspectionRayModel.has_value() && displayedBefore == &*_inspectionRayModel;

    if (op == BooleanOp::None || op == BooleanOp::Probe)
    {
        // Do not revert to the original cast; leave the last result in place.
        _rayModel = _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
    }
    else if (op == BooleanOp::Inspection)
    {
        // Prefer shelled / cut boolean stock so Inspection matches Subtraction.
        // Fall back to the pristine cast when no boolean session exists yet.
        const RayModel* baseStock =
            _booleanRayModel ? &*_booleanRayModel : _sourceRayModel;
        const bool showingBase = displayedBefore == baseStock;

        if (!_cutSweep || _cutSweep->empty())
        {
            _rayModel = nullptr;
            _inspectionRayModel.reset();
            _inspectionPrevAabb = {};
            _rayModel = baseStock;
            raysMutated = !showingBase;
        }
        else
        {
            // Fresh stock every move: clone current session stock, then subtract
            // the cutter at this pose. Do not touch _booleanRayModel.
            _rayModel = nullptr;
            const auto cloneStart = ProfileClock::now();
            _inspectionRayModel = baseStock->clone();
            cloneMs = millisSince(cloneStart);

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
            // Patch when the previous display was already base stock
            // (plus at most the last preview hole). Accumulated cuts need a
            // packed refill so old holes do not linger in the splat cache.
            haveDirtyRegion = currentDirty.valid() && (wasInspecting || showingBase);
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

    if (usesSplatView(_viewMode) && _rayModel && haveDirtyRegion && !_splatCache.empty())
    {
        _splatCache.setPointRenderMode(_viewMode == ViewMode::Disk ? PointRenderMode::HardDiskWithAA
                                                                   : PointRenderMode::Gaussian);
        const int stride = displayStride();
        const int cutStride = cutFaceStride();
        const SplatViewCull viewCull = splatViewCull();
        const auto patchStart = ProfileClock::now();
        const auto radii = splatRadii(*_rayModel, stride);
        const SplatStyle style = splatStyle();
        const bool skipCutSplats = skipCutSplatEnds(_splatCache.hasCutFace());
        PatchResult patched = PatchResult::Ok;
        if (restoreAabb.valid())
        {
            patched = _splatCache.updateRegion(*_rayModel, restoreAabb, stride, radii, style,
                                               skipCutSplats, viewCull);
            if (patched == PatchResult::Ok && dirtyModelAabb.valid())
                patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride, radii,
                                                   style, skipCutSplats, viewCull);
        }
        else
        {
            patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride, radii, style,
                                               skipCutSplats, viewCull);
        }
        if (patched == PatchResult::OutOfSpace)
        {
            // Grow is in-place now; retry the same dirty window. Packed rebuild
            // only if the sample grid itself changed (LayoutChanged below).
            if (restoreAabb.valid())
            {
                patched = _splatCache.updateRegion(*_rayModel, restoreAabb, stride, radii, style,
                                                   skipCutSplats, viewCull);
                if (patched == PatchResult::Ok && dirtyModelAabb.valid())
                    patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride, radii,
                                                       style, skipCutSplats, viewCull);
            }
            else
            {
                patched = _splatCache.updateRegion(*_rayModel, dirtyModelAabb, stride, radii,
                                                   style, skipCutSplats, viewCull);
            }
        }
        if (patched == PatchResult::Ok)
        {
            const double splatMs = millisSince(patchStart);
            const auto sectionStart = ProfileClock::now();
            if (op == BooleanOp::Inspection)
                syncInspectionSectionGrid(dirtyModelAabb);
            else
                syncCutFaceOverlay(_splatCache, *_rayModel, cutStride, splatStyle().toolColor, op,
                                   dirtyModelAabb, /*allowPatch=*/true);
            const double sectionMs = millisSince(sectionStart);
            logCutProfile(booleanMs, "patch", splatMs, dirtyModelAabb, cloneMs, sectionMs);
            if (_splatCache.gpuNeedsCompile() || !splatOnScreen())
                presentSplatCache();
            else if (_viewer)
                _viewer->request();
            return;
        }
        if (_profiling)
            std::printf("  splat patch failed (%s) after %.2f ms\n",
                        toString(patched), millisSince(patchStart));
        if (patched != PatchResult::LayoutChanged)
        {
            if (_splatCache.gpuNeedsCompile() || !splatOnScreen())
                presentSplatCache();
            else if (_viewer)
                _viewer->request();
            return;
        }
    }

    if (usesRayModel(_viewMode))
    {
        if (!_rayModel || _rayModel->rayCount() == 0)
        {
            _splatCache.clear();
            if (_current) attach(createNode(*_current), true);
            else if (_viewer) _viewer->request();
            logCutProfile(booleanMs, "no-draw", 0.0, dirtyModelAabb, cloneMs);
            return;
        }

        const auto rebuildStart = ProfileClock::now();
        if (usesSplatView(_viewMode))
        {
            _splatCache.setPointRenderMode(_viewMode == ViewMode::Disk
                                               ? PointRenderMode::HardDiskWithAA
                                               : PointRenderMode::Gaussian);
            const int stride = displayStride();
            const int cutStride = cutFaceStride();
            const SplatViewCull viewCull = splatViewCull();
            const bool skipCutSplats = skipCutSplatEnds(_splatCache.hasCutFace());
            _splatCache.rebuild(*_rayModel, stride, splatRadii(*_rayModel, stride), splatStyle(),
                                skipCutSplats, viewCull);
            const double splatMs = millisSince(rebuildStart);
            const auto sectionStart = ProfileClock::now();
            if (op == BooleanOp::Inspection)
                syncInspectionSectionGrid(dirtyModelAabb);
            else
                syncCutFaceOverlay(_splatCache, *_rayModel, cutStride, splatStyle().toolColor, op,
                                   dirtyModelAabb, /*allowPatch=*/false);
            const double sectionMs = millisSince(sectionStart);
            presentSplatCache();
            logCutProfile(booleanMs, "rebuild", splatMs, dirtyModelAabb, cloneMs, sectionMs);
        }
        else
        {
            rebuild();
            logCutProfile(booleanMs, "rebuild", millisSince(rebuildStart), dirtyModelAabb,
                          cloneMs);
        }
    }
    else
    {
        logCutProfile(booleanMs, "no-draw", 0.0, dirtyModelAabb, cloneMs);
        if (_viewer) _viewer->request();
    }
}

void RenderManager::logCutProfile(double booleanMs, const char* drawPath, double drawMs,
                                  const BoundingBox& dirtyModelAabb, double cloneMs,
                                  double sectionMs)
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
        _rayModel ? _rayModel->lastDirtyCellCount() : 0;
    const std::size_t sweepTris = _cutSweep ? _cutSweep->mesh().triangles.size() : 0;

    std::printf("cut %-4lld total %7.2f ms  clone %6.2f ms  boolean %7.2f ms  %-7s %6.2f ms"
                "  section %6.2f ms  cutMesh %s  window %9zu cells  sweep %6zu tris"
                "  intervals %8zu  pool x%.2f  splat %8zu/%-8zu %3.0f%%\n",
                _cutIndex, cloneMs + booleanMs + drawMs + sectionMs, cloneMs, booleanMs,
                drawPath, drawMs, sectionMs, cutMeshEnabled() ? "on" : "off", windowCells,
                sweepTris, liveIntervals, poolRatio, splatLive, splatCap, splatFill * 100.0);
    if (_rayModel)
    {
        const PairingStats& s = _rayModel->pairingStats();
        std::printf("  pairing  odd-rays %lld  unmatched enter %lld leave %lld  slivers %lld\n",
                    s.oddHitRays, s.unmatchedEnter, s.unmatchedLeave, s.sliversDropped);
    }
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

bool RenderManager::attach(vsg::ref_ptr<vsg::Node> node, bool replaceExisting)
{
    if (!node) return false;

    // The viewer is already running, so the new subgraph has to be compiled
    // before it can be recorded. compileManager is set up by Viewer::compile().
    if (_viewer && _viewer->compileManager)
    {
        auto compileResult = _viewer->compileManager->compile(node);
        // Recording a subgraph whose compile failed dereferences BufferInfos
        // that carry no vk buffer. Leave it detached and report the failure so
        // the caller keeps its dirty flag and retries on a later frame.
        if (!compileResult) return false;
        vsg::updateViewer(*_viewer, compileResult);
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
    return true;
}

} // namespace app
