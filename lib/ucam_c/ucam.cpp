#include "ucam.h"

#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>

#include <vsg/all.h>
#include <vsgQt/Viewer.h>
#include <vsgQt/Window.h>

#include "BRep.h"
#include "BooleanOp.h"
#include "BoundingBox.h"
#include "Parameter.h"
#include "RayGrid.h"
#include "RayModel.h"
#include "RenderManager.h"
#include "SweptVolume.h"
#include "ToolGeometry.h"
#include "ToolType.h"
#include "TriangleMesh.h"
#include "ViewMode.h"

namespace
{

thread_local std::string g_lastError;

void setError(const std::string& message)
{
    g_lastError = message;
}

void clearError()
{
    g_lastError.clear();
}

app::ToolType toToolType(int32_t type)
{
    switch (type)
    {
    case UCAM::Boolean::TOOL_BULL_NOSE: return app::ToolType::BullNose;
    case UCAM::Boolean::TOOL_FLAT_NOSE: return app::ToolType::FlatNose;
    case UCAM::Boolean::TOOL_BALL_NOSE: return app::ToolType::BallNose;
    case UCAM::Boolean::TOOL_SPHERE: return app::ToolType::Sphere;
    case UCAM::Boolean::TOOL_GRINDING_WHEEL: return app::ToolType::GrindingWheel;
    default: return app::ToolType::None;
    }
}

app::BooleanOp toBooleanOp(int32_t op)
{
    switch (op)
    {
    case UCAM::Boolean::OP_PROBE: return app::BooleanOp::Probe;
    case UCAM::Boolean::OP_SUBTRACTION: return app::BooleanOp::Subtraction;
    case UCAM::Boolean::OP_UNION: return app::BooleanOp::Union;
    case UCAM::Boolean::OP_INSPECTION: return app::BooleanOp::Inspection;
    default: return app::BooleanOp::None;
    }
}

UCAM::status rasterFrame(const app::RayModel& model, uint8_t* rgba8, int32_t width,
                         int32_t height)
{
    const auto chainLock = model.lockChains();
    const app::BoundingBox& box = model.bounds();
    if (!box.valid())
    {
        setError("frame: stock bounds are empty");
        return UCAM::ERR_FAILED;
    }

    const double minX = box.min()[0];
    const double minY = box.min()[1];
    const double extX = box.extent(0);
    const double extY = box.extent(1);
    const double sx = (extX > 1.0e-12) ? static_cast<double>(width - 1) / extX : 1.0;
    const double sy = (extY > 1.0e-12) ? static_cast<double>(height - 1) / extY : 1.0;

    std::memset(rgba8, 20, static_cast<size_t>(width) * static_cast<size_t>(height) * 4);

    auto plot = [&](double x, double y, uint8_t r, uint8_t g, uint8_t b) {
        const int px = static_cast<int>((x - minX) * sx + 0.5);
        const int py = static_cast<int>((y - minY) * sy + 0.5);
        if (px < 0 || py < 0 || px >= width || py >= height) return;
        const int iy = height - 1 - py;
        uint8_t* p = rgba8 + (static_cast<size_t>(iy) * static_cast<size_t>(width) +
                              static_cast<size_t>(px)) *
                                 4;
        p[0] = r;
        p[1] = g;
        p[2] = b;
        p[3] = 255;
    };

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const app::RayGrid* grid = model.grid(axis);
        if (!grid) continue;
        const std::size_t u = (axis + 1) % 3;
        const std::size_t v = (axis + 2) % 3;
        for (std::uint32_t iu = 0; iu < grid->width; ++iu)
        {
            for (std::uint32_t iv = 0; iv < grid->height; ++iv)
            {
                const app::RaySlot& slot = grid->at(iu, iv);
                if (slot.empty()) continue;
                const double su = grid->sampleU(iu);
                const double sv = grid->sampleV(iv);
                for (const auto& span : grid->pool.span(slot))
                {
                    if (!span.hasSolidLength()) continue;
                    double p0[3]{};
                    double p1[3]{};
                    p0[axis] = grid->fromTick(span.begin);
                    p1[axis] = grid->fromTick(span.end);
                    p0[u] = p1[u] = su;
                    p0[v] = p1[v] = sv;
                    const bool cut = span.cutBegin() || span.cutEnd();
                    plot(p0[0], p0[1], cut ? 242 : 122, cut ? 89 : 128, cut ? 26 : 138);
                    plot(p1[0], p1[1], cut ? 242 : 122, cut ? 89 : 128, cut ? 26 : 138);
                }
            }
        }
    }
    return UCAM::OK;
}

struct Env
{
    std::vector<std::string> argStore;
    std::vector<char*> argv;
    int argc = 0;
    std::unique_ptr<QApplication> ownedApp;
    vsg::ref_ptr<vsg::WindowTraits> traits;
    vsg::ref_ptr<vsg::Options> options;
    bool inited = false;
};

Env g_env;

} // namespace

namespace UCAM
{

int32_t abi_version()
{
    return ABI_VERSION;
}

const char* last_error()
{
    return g_lastError.c_str();
}

namespace Boolean
{

struct session
{
    int unused = 0;
};

struct stock
{
    app::BRep brep;
    std::optional<app::RayModel> rays;
};

struct sweep
{
    app::ToolType type = app::ToolType::FlatNose;
    float radius = 0.05f;
    float length = 0.14f;
    std::vector<app::ToolPose> poses;
    app::SweptVolume volume;
    bool built = false;
};

status session_create(session** out)
{
    clearError();
    if (!out)
    {
        setError("session_create: out is null");
        return ERR_INVALID_ARG;
    }
    *out = new session();
    return OK;
}

void session_destroy(session* session)
{
    delete session;
}

status stock_from_triangles(session* session,
                            const float* xyz,
                            uint32_t vertex_count,
                            const uint32_t* indices,
                            uint32_t triangle_count,
                            stock** out)
{
    clearError();
    if (!session || !xyz || !indices || !out || vertex_count < 3 || triangle_count < 1)
    {
        setError("stock_from_triangles: invalid arguments");
        return ERR_INVALID_ARG;
    }
    try
    {
        app::TriangleMesh mesh;
        mesh.name = "ucam_stock";
        mesh.triangles.reserve(triangle_count);
        for (uint32_t t = 0; t < triangle_count; ++t)
        {
            const uint32_t i0 = indices[t * 3 + 0];
            const uint32_t i1 = indices[t * 3 + 1];
            const uint32_t i2 = indices[t * 3 + 2];
            if (i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count)
            {
                setError("stock_from_triangles: index out of range");
                return ERR_INVALID_ARG;
            }
            app::MeshTriangle tri;
            tri.v0 = vsg::vec3(xyz[i0 * 3 + 0], xyz[i0 * 3 + 1], xyz[i0 * 3 + 2]);
            tri.v1 = vsg::vec3(xyz[i1 * 3 + 0], xyz[i1 * 3 + 1], xyz[i1 * 3 + 2]);
            tri.v2 = vsg::vec3(xyz[i2 * 3 + 0], xyz[i2 * 3 + 1], xyz[i2 * 3 + 2]);
            const vsg::vec3 e1 = tri.v1 - tri.v0;
            const vsg::vec3 e2 = tri.v2 - tri.v0;
            const vsg::vec3 n = vsg::cross(e1, e2);
            const float nLen = vsg::length(n);
            tri.normal = (nLen > 0.0f) ? n / nLen : vsg::vec3(0.0f, 0.0f, 1.0f);
            mesh.triangles.push_back(tri);
        }
        auto stockPtr = std::make_unique<stock>();
        stockPtr->brep = app::BRep::fromTriangles(mesh);
        if (stockPtr->brep.faceCount() == 0)
        {
            setError("stock_from_triangles: mesh produced no faces");
            return ERR_FAILED;
        }
        *out = stockPtr.release();
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

void stock_destroy(stock* stock)
{
    delete stock;
}

status stock_cast(stock* stock, double res_x, double res_y, double res_z)
{
    clearError();
    if (!stock || !(res_x > 0.0) || !(res_y > 0.0) || !(res_z > 0.0))
    {
        setError("stock_cast: invalid stock or resolution");
        return ERR_INVALID_ARG;
    }
    try
    {
        const app::Point3d requested{res_x, res_y, res_z};
        const app::Point3d res = app::RayModel::finestCastableResolution(stock->brep, requested);
        stock->rays = app::RayModel::fromBRep(stock->brep, res);
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

uint64_t stock_ray_count(const stock* stock)
{
    if (!stock || !stock->rays) return 0;
    return static_cast<uint64_t>(stock->rays->rayCount());
}

uint64_t stock_dirty_cell_count(const stock* stock)
{
    if (!stock || !stock->rays) return 0;
    return static_cast<uint64_t>(stock->rays->lastDirtyCellCount());
}

status stock_export_triangles(const stock* stock,
                              float* xyz,
                              uint32_t* vertex_count,
                              uint32_t* indices,
                              uint32_t* triangle_count)
{
    clearError();
    if (!stock || !vertex_count || !triangle_count)
    {
        setError("stock_export_triangles: invalid arguments");
        return ERR_INVALID_ARG;
    }
    const uint32_t vcount = static_cast<uint32_t>(stock->brep.vertexCount());
    const uint32_t tcount = static_cast<uint32_t>(stock->brep.faceCount());
    if (!xyz || !indices)
    {
        *vertex_count = vcount;
        *triangle_count = tcount;
        return OK;
    }
    if (*vertex_count < vcount || *triangle_count < tcount)
    {
        setError("stock_export_triangles: buffer too small");
        *vertex_count = vcount;
        *triangle_count = tcount;
        return ERR_INVALID_ARG;
    }
    const auto& verts = stock->brep.vertices();
    for (uint32_t i = 0; i < vcount; ++i)
    {
        xyz[i * 3 + 0] = verts[i].x;
        xyz[i * 3 + 1] = verts[i].y;
        xyz[i * 3 + 2] = verts[i].z;
    }
    const auto& offsets = stock->brep.faceOffsets();
    const auto& fverts = stock->brep.faceVertices();
    for (uint32_t f = 0; f < tcount; ++f)
    {
        const uint32_t begin = offsets[f];
        indices[f * 3 + 0] = fverts[begin];
        indices[f * 3 + 1] = fverts[begin + 1];
        indices[f * 3 + 2] = fverts[begin + 2];
    }
    *vertex_count = vcount;
    *triangle_count = tcount;
    return OK;
}

status sweep_begin(session* session,
                   int32_t tool_type,
                   double radius,
                   double length,
                   sweep** out)
{
    clearError();
    if (!session || !out || !(radius > 0.0) || !(length > 0.0))
    {
        setError("sweep_begin: invalid arguments");
        return ERR_INVALID_ARG;
    }
    const app::ToolType type = toToolType(tool_type);
    if (type == app::ToolType::None)
    {
        setError("sweep_begin: tool_type cannot be NONE");
        return ERR_INVALID_ARG;
    }
    auto sweepPtr = std::make_unique<sweep>();
    sweepPtr->type = type;
    sweepPtr->radius = static_cast<float>(radius);
    sweepPtr->length = static_cast<float>(length);
    *out = sweepPtr.release();
    return OK;
}

status sweep_add_pose(sweep* sweep,
                      double x, double y, double z,
                      double dx, double dy, double dz)
{
    clearError();
    if (!sweep)
    {
        setError("sweep_add_pose: sweep is null");
        return ERR_INVALID_ARG;
    }
    app::ToolPose pose;
    pose.position = vsg::dvec3(x, y, z);
    const vsg::dvec3 dir(dx, dy, dz);
    const double len = vsg::length(dir);
    pose.direction = (len > 1.0e-12) ? dir / len : vsg::dvec3(0.0, 0.0, 1.0);
    sweep->poses.push_back(pose);
    sweep->built = false;
    return OK;
}

status sweep_end(sweep* sweep)
{
    clearError();
    if (!sweep)
    {
        setError("sweep_end: sweep is null");
        return ERR_INVALID_ARG;
    }
    if (sweep->poses.size() < 2)
    {
        setError("sweep_end: need at least two poses");
        return ERR_INVALID_ARG;
    }
    try
    {
        sweep->volume.clear();
        sweep->volume.appendPath(sweep->type, sweep->radius, sweep->length, sweep->poses);
        sweep->built = !sweep->volume.empty();
        if (!sweep->built)
        {
            setError("sweep_end: sweep produced no triangles");
            return ERR_FAILED;
        }
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

void sweep_destroy(sweep* sweep)
{
    delete sweep;
}

status apply(stock* stock, sweep* sweep, int32_t op)
{
    clearError();
    if (!stock || !sweep)
    {
        setError("apply: stock or sweep is null");
        return ERR_INVALID_ARG;
    }
    if (!stock->rays)
    {
        setError("apply: stock has not been cast");
        return ERR_INVALID_ARG;
    }
    if (!sweep->built)
    {
        setError("apply: call sweep_end first");
        return ERR_INVALID_ARG;
    }
    const app::BooleanOp bop = toBooleanOp(op);
    if (!app::appliesBoolean(bop))
    {
        setError("apply: op must be SUBTRACTION, UNION, or INSPECTION");
        return ERR_INVALID_ARG;
    }
    try
    {
        stock->rays->booleanInPlace(sweep->volume, bop, vsg::dmat4{});
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

} // namespace Boolean

namespace Graphics
{

struct window
{
    QMainWindow* main = nullptr;
    vsgQt::Window* vsgWindow = nullptr;
    vsg::ref_ptr<vsgQt::Viewer> viewer;
    vsg::ref_ptr<vsg::WindowTraits> traits;
    bool shown = false;
};

struct view
{
    window* win = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    bool offscreen = false;
    Boolean::stock* stock = nullptr;
    app::ToolType toolType = app::ToolType::None;
    double radius = 0.05;
    double length = 0.14;
    app::ToolPose pose{};
    bool disk = true;
    bool cutMesh = false;
    vsg::ref_ptr<vsg::Group> scene;
    std::shared_ptr<app::RenderManager> renderManager;
    vsg::ref_ptr<vsg::Camera> camera;
    bool compiled = false;
    bool stockShown = false;
};

void applyViewFlags(view* view)
{
    if (!view) return;
    app::Parameter::instance().setViewMode(view->disk ? app::ViewMode::Disk : app::ViewMode::RayGS);
    app::Parameter::instance().setCutMeshDisplay(view->cutMesh);
    if (view->renderManager)
    {
        view->renderManager->setViewMode(view->disk ? app::ViewMode::Disk : app::ViewMode::RayGS);
        view->renderManager->refreshCutMeshDisplay();
    }
}

void applyTool(view* view)
{
    if (!view || !view->renderManager) return;
    app::Parameter::instance().setToolType(view->toolType);
    app::Parameter::instance().setToolRadius(view->radius);
    app::Parameter::instance().setToolLength(view->length);
    view->renderManager->setToolType(view->toolType);
    view->renderManager->updateToolGeometry();
    if (view->toolType != app::ToolType::None)
        view->renderManager->setToolPose(view->pose.position, view->pose.direction);
}

status init(int* argc, char** argv)
{
    clearError();
    if (g_env.inited) return OK;

    if (QApplication::instance())
    {
        g_env.traits = vsg::WindowTraits::create();
        g_env.traits->windowTitle = "UCAM";
        g_env.traits->width = 800;
        g_env.traits->height = 600;
        g_env.options = vsg::Options::create();
        g_env.inited = true;
        return OK;
    }

    if (!argc || !argv)
    {
        setError("init: argc/argv are required when no QApplication exists");
        return ERR_INVALID_ARG;
    }

    g_env.argStore.clear();
    g_env.argv.clear();
    g_env.argc = *argc;
    g_env.argStore.reserve(static_cast<size_t>(*argc));
    for (int i = 0; i < *argc; ++i)
        g_env.argStore.emplace_back(argv[i] ? argv[i] : "");
    g_env.argv.reserve(g_env.argStore.size());
    for (auto& s : g_env.argStore)
        g_env.argv.push_back(s.data());
    g_env.argc = static_cast<int>(g_env.argv.size());

    try
    {
        g_env.ownedApp = std::make_unique<QApplication>(g_env.argc, g_env.argv.data());
        g_env.traits = vsg::WindowTraits::create();
        g_env.traits->windowTitle = "UCAM";
        g_env.traits->width = 800;
        g_env.traits->height = 600;
        g_env.options = vsg::Options::create();
        g_env.options->paths = vsg::getEnvPaths("VSG_FILE_PATH");
        g_env.inited = true;
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

void shutdown()
{
    g_env.ownedApp.reset();
    g_env.traits = nullptr;
    g_env.options = nullptr;
    g_env.inited = false;
    g_env.argStore.clear();
    g_env.argv.clear();
    g_env.argc = 0;
}

status window_create(int32_t width, int32_t height, const char* title, window** out)
{
    clearError();
    if (!out || width < 1 || height < 1)
    {
        setError("window_create: invalid size");
        return ERR_INVALID_ARG;
    }
    if (!g_env.inited || !QApplication::instance())
    {
        setError("window_create: call Graphics::init first");
        return ERR_INVALID_ARG;
    }

    try
    {
        auto win = std::make_unique<window>();
        win->traits = vsg::WindowTraits::create();
        win->traits->width = static_cast<uint32_t>(width);
        win->traits->height = static_cast<uint32_t>(height);
        win->traits->windowTitle = title ? title : "UCAM";

        win->viewer = vsgQt::Viewer::create();
        win->vsgWindow = new vsgQt::Window(win->viewer, win->traits);
        win->vsgWindow->setTitle(win->traits->windowTitle.c_str());

        win->main = new QMainWindow();
        win->main->setWindowTitle(QString::fromUtf8(win->traits->windowTitle.c_str()));
        auto* container = QWidget::createWindowContainer(win->vsgWindow, win->main);
        container->setMinimumSize(320, 240);
        win->main->setCentralWidget(container);
        win->main->resize(width, height);
        win->main->show();
        QApplication::instance()->processEvents();
        win->shown = true;

        *out = win.release();
        return OK;
    }
    catch (const vsg::Exception& ex)
    {
        setError(ex.message);
        return ERR_FAILED;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

void window_destroy(window* window)
{
    if (!window) return;
    delete window->main;
    window->main = nullptr;
    window->vsgWindow = nullptr;
    window->viewer = nullptr;
    delete window;
}

status create(window* window, view** out)
{
    clearError();
    if (!window || !out)
    {
        setError("create: invalid arguments");
        return ERR_INVALID_ARG;
    }
    if (!window->vsgWindow || !window->viewer || !window->main)
    {
        setError("create: window is not live");
        return ERR_INVALID_ARG;
    }

    try
    {
        auto v = std::make_unique<view>();
        v->win = window;
        v->width = static_cast<int32_t>(window->traits->width);
        v->height = static_cast<int32_t>(window->traits->height);
        v->scene = vsg::Group::create();
        v->renderManager = std::make_shared<app::RenderManager>(window->viewer, v->scene,
                                                                g_env.options);
        v->renderManager->setFitToUnitBox(false);

        window->vsgWindow->initializeWindow();
        if (!window->vsgWindow->windowAdapter)
        {
            setError("create: failed to create VSG window adapter");
            return ERR_FAILED;
        }
        if (!window->traits->device)
            window->traits->device = window->vsgWindow->windowAdapter->getOrCreateDevice();

        vsg::ComputeBounds computeBounds;
        v->scene->accept(computeBounds);
        vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
        double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
        const double viewRadius = (radius > 1.0e-6) ? radius : 1.0;
        const uint32_t width = window->traits->width;
        const uint32_t height = window->traits->height;

        auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -viewRadius * 3.5, 0.0),
                                          centre, vsg::dvec3(0.0, 0.0, 1.0));
        auto perspective = vsg::Perspective::create(
            30.0, static_cast<double>(width) / static_cast<double>(height),
            0.001 * viewRadius, viewRadius * 100.0);
        v->camera = vsg::Camera::create(perspective, lookAt,
                                        vsg::ViewportState::create(VkExtent2D{width, height}));

        auto trackball = vsg::Trackball::create(v->camera);
        trackball->supportsThrow = false;
        trackball->addWindow(*window->vsgWindow);
        window->viewer->addEventHandler(trackball);
        window->viewer->addEventHandler(vsg::CloseHandler::create(window->viewer));

        auto renderGraph = vsg::createRenderGraphForView(*window->vsgWindow, v->camera, v->scene);
        auto commandGraph = vsg::CommandGraph::create(*window->vsgWindow);
        commandGraph->addChild(renderGraph);
        window->viewer->addRecordAndSubmitTaskAndPresentation({commandGraph});

        v->renderManager->configureRayBudgets(window->traits->device);
        v->renderManager->setCamera(v->camera);
        window->viewer->compile();
        v->compiled = true;

        *out = v.release();
        return OK;
    }
    catch (const vsg::Exception& ex)
    {
        setError(ex.message);
        return ERR_FAILED;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

void destroy(view* view)
{
    delete view;
}

status set_stock(view* view, Boolean::stock* stock)
{
    clearError();
    if (!view || !stock)
    {
        setError("set_stock: invalid arguments");
        return ERR_INVALID_ARG;
    }
    if (!stock->rays)
    {
        setError("set_stock: stock has not been cast");
        return ERR_INVALID_ARG;
    }
    view->stock = stock;
    if (!view->renderManager) return OK;

    try
    {
        if (!view->stockShown)
        {
            view->renderManager->showBRep(stock->brep);
            view->stockShown = true;
        }
        view->renderManager->setRayModel(stock->rays->clone());
        applyViewFlags(view);
        applyTool(view);
        if (view->win && view->win->viewer) view->win->viewer->request();
        return OK;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

status set_tool(view* view, int32_t tool_type, double radius, double length)
{
    clearError();
    if (!view || !(radius > 0.0) || !(length > 0.0))
    {
        setError("set_tool: invalid arguments");
        return ERR_INVALID_ARG;
    }
    view->toolType = toToolType(tool_type);
    view->radius = radius;
    view->length = length;
    app::Parameter::instance().setToolType(view->toolType);
    app::Parameter::instance().setToolRadius(radius);
    app::Parameter::instance().setToolLength(length);
    applyTool(view);
    return OK;
}

status commit_pose(view* view, double x, double y, double z, double dx, double dy, double dz)
{
    clearError();
    if (!view)
    {
        setError("commit_pose: view is null");
        return ERR_INVALID_ARG;
    }
    view->pose.position = vsg::dvec3(x, y, z);
    const vsg::dvec3 dir(dx, dy, dz);
    const double len = vsg::length(dir);
    view->pose.direction = (len > 1.0e-12) ? dir / len : vsg::dvec3(0.0, 0.0, 1.0);
    if (view->renderManager && view->toolType != app::ToolType::None)
        view->renderManager->setToolPose(view->pose.position, view->pose.direction);
    if (view->win && view->win->viewer) view->win->viewer->request();
    return OK;
}

status set_flags(view* view, int32_t disk, int32_t cut_mesh)
{
    clearError();
    if (!view)
    {
        setError("set_flags: view is null");
        return ERR_INVALID_ARG;
    }
    view->disk = disk != 0;
    view->cutMesh = cut_mesh != 0;
    applyViewFlags(view);
    return OK;
}

status poll(window* window)
{
    clearError();
    if (!window || !window->main || !window->viewer)
    {
        setError("poll: window is null");
        return ERR_INVALID_ARG;
    }
    if (!window->main->isVisible())
    {
        setError("poll: window was closed");
        return ERR_FAILED;
    }
    QApplication::instance()->processEvents();
    if (!window->main->isVisible())
    {
        setError("poll: window was closed");
        return ERR_FAILED;
    }
    try
    {
        window->viewer->render();
        return OK;
    }
    catch (const vsg::Exception& ex)
    {
        setError(ex.message);
        return ERR_FAILED;
    }
    catch (const std::exception& ex)
    {
        setError(ex.what());
        return ERR_FAILED;
    }
}

status run(window* window)
{
    clearError();
    if (!window || !window->main || !window->viewer)
    {
        setError("run: window is null");
        return ERR_INVALID_ARG;
    }
    window->main->show();
    window->viewer->continuousUpdate = true;
    window->viewer->setInterval(16);
    const int code = QApplication::exec();
    if (code != 0)
    {
        setError("run: QApplication::exec failed");
        return ERR_FAILED;
    }
    return OK;
}

status create_offscreen(int32_t width, int32_t height, view** out)
{
    clearError();
    if (!out || width < 1 || height < 1)
    {
        setError("create_offscreen: invalid size");
        return ERR_INVALID_ARG;
    }
    auto v = std::make_unique<view>();
    v->offscreen = true;
    v->width = width;
    v->height = height;
    *out = v.release();
    return OK;
}

status frame(view* view, uint8_t* rgba8, int32_t width, int32_t height)
{
    clearError();
    if (!view || !rgba8 || width < 1 || height < 1)
    {
        setError("frame: invalid arguments");
        return ERR_INVALID_ARG;
    }
    if (width != view->width || height != view->height)
    {
        setError("frame: buffer size does not match the view");
        return ERR_INVALID_ARG;
    }
    if (!view->stock || !view->stock->rays)
    {
        setError("frame: set a cast stock first");
        return ERR_INVALID_ARG;
    }
    return rasterFrame(*view->stock->rays, rgba8, width, height);
}

} // namespace Graphics

} // namespace UCAM
