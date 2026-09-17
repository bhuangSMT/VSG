// vsg_qt_cube
//
// Creates a Qt main window (QMainWindow) with dockable Controls and Simulation
// panels around a VulkanSceneGraph rendering surface embedded via vsgQt. Geometry is held as a BRep and drawn by the RenderManager: the
// scene starts with a capped cylinder (r=1, L=10), and STL or 3MF files can be imported at
// runtime via the panel's import buttons. The panel's view mode selector switches
// between facet and wireframe rendering of that BRep.
//
// Controls: left-drag to rotate, right-drag / wheel to zoom, middle-drag to pan.
// Right-click (no drag) opens the Operation context menu (exit data collection).

#include <vsg/all.h>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QStyleFactory>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QInputDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QScrollArea>
#include <QtGui/QColor>
#include <QtGui/QCursor>
#include <QtGui/QFont>
#include <QtGui/QKeySequence>
#include <QtGui/QMouseEvent>
#include <QtGui/QPalette>
#include <QtGui/QWindow>
#include <QtGui/QShortcut>
#include <QtGui/QStandardItemModel>

#include <vsgQt/Window.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <vector>
#include <iostream>
#include <memory>
#include <optional>

#include "BRep.h"
#include "BoundingBox.h"
#include "ControlCube.h"
#include "Parameter.h"
#include "RayModel.h"
#include "RenderManager.h"
#include "SimulationPanel.h"
#include "StlImporter.h"
#include "ThreeMfImporter.h"
#include "ToolManagerDialog.h"
#include "ToolTracker.h"
#include "ToolType.h"
#include "UcamDebug.h"

namespace
{

// Chooses the reader from the file's extension, so the two import buttons and
// the --import option all end up in the same place and either format can be
// typed into either dialog.
app::TriangleMesh importMesh(const std::string& path)
{
    const auto dot = path.rfind('.');
    std::string extension = (dot == std::string::npos) ? std::string() : path.substr(dot + 1);
    for (char& c : extension) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (extension == "stl") return app::StlImporter().import(path);
    if (extension == "3mf") return app::ThreeMfImporter().import(path);

    throw std::runtime_error("Unsupported model format (expected .stl or .3mf): " + path);
}

// The startup geometry: a 1x1x1 cube centred on the origin, expressed as a
// BRep so it travels the same path as imported models and therefore responds
// to the view mode selector. Cylinder along +X (horizontal), radius 1, length 10,
// both end caps.
app::BRep createCylinderBRep()
{
    constexpr float radius = 1.0f;
    constexpr float length = 10.0f;
    constexpr int slices = 64;
    constexpr float pi = 3.14159265358979323846f;
    const float x0 = -0.5f * length;
    const float x1 = 0.5f * length;

    app::TriangleMesh mesh;
    mesh.name = "cylinder";
    mesh.triangles.reserve(static_cast<std::size_t>(slices * 4));

    std::vector<vsg::vec3> end0;
    std::vector<vsg::vec3> end1;
    end0.reserve(static_cast<std::size_t>(slices));
    end1.reserve(static_cast<std::size_t>(slices));
    for (int i = 0; i < slices; ++i)
    {
        const float a = (2.0f * pi * static_cast<float>(i)) / static_cast<float>(slices);
        const float y = radius * std::cos(a);
        const float z = radius * std::sin(a);
        end0.push_back(vsg::vec3(x0, y, z));
        end1.push_back(vsg::vec3(x1, y, z));
    }

    const vsg::vec3 c0(x0, 0.0f, 0.0f);
    const vsg::vec3 c1(x1, 0.0f, 0.0f);
    const vsg::vec3 n0(-1.0f, 0.0f, 0.0f);
    const vsg::vec3 n1(1.0f, 0.0f, 0.0f);

    auto addTri = [&](const vsg::vec3& a, const vsg::vec3& b, const vsg::vec3& c,
                      const vsg::vec3& n) {
        app::MeshTriangle t;
        t.v0 = a;
        t.v1 = b;
        t.v2 = c;
        t.normal = n;
        mesh.triangles.push_back(t);
    };

    for (int i = 0; i < slices; ++i)
    {
        const int j = (i + 1) % slices;
        // Caps (outward along ±X).
        addTri(c0, end0[i], end0[j], n0);
        addTri(c1, end1[j], end1[i], n1);
        // Side wall.
        const vsg::vec3 n = vsg::normalize(vsg::vec3(0.0f, end0[i].y, end0[i].z));
        addTri(end0[i], end1[i], end1[j], n);
        addTri(end0[i], end1[j], end0[j], n);
    }

    return app::BRep::fromTriangles(mesh);
}

// After the QWindow is embedded and shown, create the Vulkan surface and
// attach a camera/trackball/command graph that renders vsg_scene. Returns the
// trackball and reports the camera and initial viewpoint so the UI can reset
// the view and the tool tracker can pick against the model.
vsg::ref_ptr<vsg::Trackball> initializeViewer(vsgQt::Window* window,
                                              vsg::ref_ptr<vsgQt::Viewer> viewer,
                                              vsg::ref_ptr<vsg::WindowTraits> traits,
                                              vsg::ref_ptr<vsg::Node> vsg_scene,
                                              vsg::ref_ptr<vsg::Options> options,
                                              vsg::ref_ptr<vsg::Camera>& out_camera,
                                              vsg::ref_ptr<vsg::LookAt>& out_lookAt,
                                              vsg::ref_ptr<app::ControlCube>& out_cube)
{
    window->initializeWindow();

    if (!window->windowAdapter)
        throw vsg::Exception{"Failed to create VSG window adapter.", VK_ERROR_INITIALIZATION_FAILED};

    if (!traits->device) traits->device = window->windowAdapter->getOrCreateDevice();

    vsg::ComputeBounds computeBounds;
    vsg_scene->accept(computeBounds);
    vsg::dvec3 centre = (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    double radius = vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    double nearFarRatio = 0.001;

    uint32_t width = window->traits->width;
    uint32_t height = window->traits->height;

    auto lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -radius * 3.5, 0.0),
                                      centre,
                                      vsg::dvec3(0.0, 0.0, 1.0));

    auto perspective = vsg::Perspective::create(
        30.0,
        static_cast<double>(width) / static_cast<double>(height),
        nearFarRatio * radius,
        radius * 4.5);

    auto camera = vsg::Camera::create(perspective, lookAt,
                                      vsg::ViewportState::create(VkExtent2D{width, height}));

    auto trackball = vsg::Trackball::create(camera);
    // Throw keeps rotating after release. With Continuous update off, frames
    // only run on mouse moves, so inertia advances in large jumps and feels
    // like the view is tracking the cursor. Keep orbit to button-drag only.
    trackball->supportsThrow = false;
    trackball->addWindow(*window);

    auto controlCube = app::ControlCube::create(camera, trackball, options,
                                                window->windowAdapter, window);
    // Cube clicks must win over orbit and the tool before those handlers run.
    viewer->addEventHandler(controlCube);
    viewer->addEventHandler(trackball);
    viewer->addEventHandler(app::ControlCubeLateSync::create(controlCube));

    auto renderGraph = vsg::createRenderGraphForView(*window, camera, vsg_scene);
    if (controlCube->depthClear())
        renderGraph->addChild(controlCube->depthClear());
    renderGraph->addChild(controlCube->view());

    auto commandGraph = vsg::CommandGraph::create(*window);
    commandGraph->addChild(renderGraph);
    viewer->addRecordAndSubmitTaskAndPresentation({commandGraph});

    // Snapshot the starting viewpoint for the panel's "Reset View" action.
    out_camera = camera;
    out_lookAt = vsg::LookAt::create(*lookAt);
    out_cube = controlCube;

    return trackball;
}

// Right-click (no drag) on the Vulkan view: Qt context menus never reach a
// QWindow container, and right-drag is already zoom. Treat a still click as
// the Simulation "exit data collection" menu.
class ViewportContextMenuFilter : public QObject
{
public:
    explicit ViewportContextMenuFilter(app::SimulationPanel* panel,
                                       app::ControlCube* cube,
                                       QObject* parent = nullptr) :
        QObject(parent), _panel(panel), _cube(cube)
    {
    }

protected:
    bool eventFilter(QObject* watched, QEvent* event) override
    {
        if (!_panel) return QObject::eventFilter(watched, event);

        if (event->type() == QEvent::MouseButtonPress)
        {
            const auto* mouse = static_cast<const QMouseEvent*>(event);
            if (mouse->button() == Qt::RightButton)
            {
                _rightPressed = !inControlCube(watched, mouse->pos());
                _pressPos = mouse->pos();
            }
        }
        else if (event->type() == QEvent::MouseButtonRelease)
        {
            const auto* mouse = static_cast<const QMouseEvent*>(event);
            if (mouse->button() == Qt::RightButton && _rightPressed)
            {
                _rightPressed = false;
                const QPoint delta = mouse->pos() - _pressPos;
                if (delta.manhattanLength() <= 6 && !inControlCube(watched, mouse->pos()))
                    _panel->popupExitCollectionMenu(QCursor::pos());
            }
        }
        return QObject::eventFilter(watched, event);
    }

private:
    bool inControlCube(QObject* watched, const QPoint& pos) const
    {
        if (!_cube) return false;
        const auto* win = qobject_cast<const QWindow*>(watched);
        const qreal dpr = win ? win->devicePixelRatio() : 1.0;
        const int32_t x = static_cast<int32_t>(std::lround(static_cast<qreal>(pos.x()) * dpr));
        const int32_t y = static_cast<int32_t>(std::lround(static_cast<qreal>(pos.y()) * dpr));
        return _cube->contains(x, y);
    }

    app::SimulationPanel* _panel = nullptr;
    app::ControlCube* _cube = nullptr;
    QPoint _pressPos;
    bool _rightPressed = false;
};

void useLightSpinArrows(QAbstractSpinBox* spin)
{
    // macOS Cocoa keeps native black steppers unless the widget uses Fusion.
    if (spin) spin->setStyle(QStyleFactory::create("Fusion"));
}

} // namespace

int main(int argc, char* argv[])
try
{
    QApplication application(argc, argv);

    const QColor themeColor(0x19, 0x22, 0x3b);
    const QColor textColor(0xe8, 0xee, 0xf4);
    QPalette theme = application.palette();
    theme.setColor(QPalette::Window, themeColor);
    theme.setColor(QPalette::WindowText, textColor);
    theme.setColor(QPalette::Base, QColor(0x24, 0x33, 0x52));
    theme.setColor(QPalette::AlternateBase, QColor(0x1e, 0x2a, 0x46));
    theme.setColor(QPalette::Text, textColor);
    theme.setColor(QPalette::Button, QColor(0x2a, 0x3a, 0x5c));
    theme.setColor(QPalette::ButtonText, textColor);
    theme.setColor(QPalette::Light, QColor(0x3a, 0x4c, 0x72));
    theme.setColor(QPalette::Midlight, QColor(0x2e, 0x3e, 0x60));
    theme.setColor(QPalette::Mid, QColor(0x14, 0x1b, 0x30));
    theme.setColor(QPalette::Dark, QColor(0x10, 0x16, 0x28));
    theme.setColor(QPalette::Highlight, QColor(0x3d, 0x5a, 0x8c));
    theme.setColor(QPalette::HighlightedText, textColor);
    application.setPalette(theme);
    application.setStyleSheet(QStringLiteral(
        "QMainWindow, QDialog, QDockWidget, QMenuBar, #central, #sidePanel, #simPanel {"
        "  background-color: #19223b; color: #e8eef4;"
        "}"
        "QMenuBar::item:selected { background-color: #2a3a5c; }"
        "QDockWidget::title { background: #243352; padding: 4px 8px; }"
        "QLabel, QCheckBox { background-color: transparent; color: #e8eef4; }"
        "QLineEdit, QAbstractSpinBox, QComboBox, QComboBox QAbstractItemView,"
        "QTableWidget, QTableView, QHeaderView::section, QMenu {"
        "  background-color: #243352; color: #e8eef4;"
        "}"
        "QPushButton, QComboBox, QAbstractSpinBox, QLineEdit { padding: 2px 8px; }"
        "QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {"
        "  background: #243352; border: none; width: 16px;"
        "}"
        "QAbstractSpinBox::up-arrow {"
        "  image: none; width: 0; height: 0;"
        "  border-left: 4px solid transparent;"
        "  border-right: 4px solid transparent;"
        "  border-bottom: 5px solid #e8eef4;"
        "}"
        "QAbstractSpinBox::down-arrow {"
        "  image: none; width: 0; height: 0;"
        "  border-left: 4px solid transparent;"
        "  border-right: 4px solid transparent;"
        "  border-top: 5px solid #e8eef4;"
        "}"
        "QHeaderView::section { padding: 4px 6px; }"
        "QTableWidget::item, QTableView::item { padding: 3px 4px; }"
        "QPushButton { background-color: #2a3a5c; color: #e8eef4; }"
        "QPushButton:hover { background-color: #3a4c72; }"
        "QScrollBar:vertical { background: #19223b; width: 12px; margin: 0; }"
        "QScrollBar::handle:vertical { background: #4a5c82; min-height: 32px; border-radius: 5px; }"
        "QScrollBar::handle:vertical:hover { background: #5c7098; }"
        "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
        "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }"));

    // Bring the settings store up with the application, before anything that
    // reads or writes it exists.
    app::Parameter& parameters = app::Parameter::instance();

    vsg::CommandLine arguments(&argc, argv);

    auto options = vsg::Options::create();
    options->fileCache = vsg::getEnv("VSG_FILE_CACHE");
    options->paths = vsg::getEnvPaths("VSG_FILE_PATH");

    auto windowTraits = vsg::WindowTraits::create();
    windowTraits->windowTitle = "UCAM";
    windowTraits->width = 1024;
    windowTraits->height = 768;
    windowTraits->debugLayer = arguments.read({"--debug", "-d"});
    windowTraits->apiDumpLayer = arguments.read({"--api", "-a"});
    arguments.read("--samples", windowTraits->samples);
    arguments.read({"--window", "-w"}, windowTraits->width, windowTraits->height);

    auto interval = arguments.value<int>(8, "--interval");

    std::string startupModel;
    arguments.read("--import", startupModel);

    std::string startupViewMode;
    arguments.read("--view-mode", startupViewMode);

    std::string startupTool;
    arguments.read("--tool", startupTool);

    // Rays are cast on a grid of this many steps across the model's bounding
    // box, so the sampling adapts to the model's own size.
    auto rayDivisions = arguments.value<int>(1600, "--ray-divisions");

    // Log per-cut timings to stdout so interactive slowdown is measurable.
    const bool profileCuts = arguments.read({"--profile", "-p"});

    if (arguments.errors()) return arguments.writeErrorMessages(std::cerr);

    if (rayDivisions < 1)
    {
        std::cerr << "--ray-divisions must be at least 1" << std::endl;
        return 1;
    }

    // Command-line values are settings too, so they are committed to the store
    // rather than passed around separately.
    parameters.setRayDivisions(rayDivisions);

    const bool debugUi = app::ucamDebugEnabled();
    std::optional<app::ViewMode> startupViewModeValue;
    if (!startupViewMode.empty())
    {
        if (startupViewMode == "facet")
            startupViewModeValue = app::ViewMode::Facet;
        else if (startupViewMode == "wireframe")
            startupViewModeValue = app::ViewMode::Wireframe;
        else if (startupViewMode == "ray")
            startupViewModeValue = app::ViewMode::Ray;
        else if (startupViewMode == "ray-gs")
            startupViewModeValue = app::ViewMode::RayGS;
        else
        {
            std::cerr << "unknown --view-mode '" << startupViewMode
                      << "', expected facet, wireframe, ray or ray-gs" << std::endl;
            return 1;
        }
        if (*startupViewModeValue == app::ViewMode::Ray && !debugUi)
        {
            std::cerr << "--view-mode ray requires " << app::kUcamDebugEnv << '='
                      << app::kUcamDebugKey << std::endl;
            return 1;
        }
    }

    int startupToolIndex = 0;
    if (!startupTool.empty())
    {
        if (startupTool == "none")
            startupToolIndex = 0;
        else if (startupTool == "bull-nose" || startupTool == "bull")
            startupToolIndex = 1;
        else if (startupTool == "flat-nose" || startupTool == "flat")
            startupToolIndex = 2;
        else if (startupTool == "ball-nose" || startupTool == "ball")
            startupToolIndex = 3;
        else if (startupTool == "sphere")
            startupToolIndex = 4;
        else
        {
            std::cerr << "unknown --tool '" << startupTool
                      << "', expected none, bull-nose, flat-nose, ball-nose or sphere"
                      << std::endl;
            return 1;
        }
    }

    // The RenderManager owns the contents of this root; it is populated with
    // the startup cube before the camera is framed around it.
    auto vsg_scene = vsg::Group::create();

    // Qt main window that hosts the Vulkan-rendered surface.
    auto mainWindow = new QMainWindow();
    mainWindow->setWindowTitle("UCAM");

    auto viewer = vsgQt::Viewer::create();

    // Create the vsgQt QWindow but do not initialize Vulkan yet. Embedding via
    // createWindowContainer recreates the native view; a surface created before
    // that is destroyed and the viewer then quits.
    auto window = new vsgQt::Window(viewer, windowTraits);
    window->setTitle("UCAM");

    auto renderWidget = QWidget::createWindowContainer(window, mainWindow);
    renderWidget->setMinimumSize(320, 240);

    // --- Left-hand control panel: one widget per row -----------------------
    auto panel = new QWidget();
    panel->setObjectName("sidePanel");
    panel->setAutoFillBackground(true);
    panel->setMinimumWidth(220);
    panel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

    auto column = new QVBoxLayout(panel);
    column->setContentsMargins(10, 10, 10, 10);
    column->setSpacing(10);

    auto addWidget = [&](QWidget* widget) {
        widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        column->addWidget(widget);
    };

    auto makeDivider = []() {
        auto* line = new QFrame();
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        line->setFixedHeight(2);
        line->setStyleSheet("QFrame { color: #9a9a9a; }");
        return line;
    };

    auto addDivider = [&]() {
        column->addSpacing(20);
        addWidget(makeDivider());
    };

    QFont titleFont = QApplication::font();
    titleFont.setBold(true);

    addDivider();
    auto stockTitle = new QLabel("Stock:");
    stockTitle->setFont(titleFont);
    addWidget(stockTitle);

    auto importStlButton = new QPushButton("Import STL…");
    addWidget(importStlButton);
    column->addSpacing(10);
    auto import3mfButton = new QPushButton("Import 3MF…");
    addWidget(import3mfButton);

    auto title = new QLabel("Scene Controls");
    title->setFont(titleFont);
    addWidget(title);

    auto resetButton = new QPushButton("Reset View");
    addWidget(resetButton);

    addWidget(new QLabel("View Mode"));
    auto viewModeCombo = new QComboBox();
    viewModeCombo->addItem("Facet", static_cast<int>(app::ViewMode::Facet));
    viewModeCombo->addItem("Wireframe", static_cast<int>(app::ViewMode::Wireframe));
    if (debugUi)
        viewModeCombo->addItem("Ray", static_cast<int>(app::ViewMode::Ray));
    viewModeCombo->addItem("Simulation", static_cast<int>(app::ViewMode::RayGS));
    addWidget(viewModeCombo);

    int startupViewModeIndex = 0;
    if (startupViewModeValue)
    {
        const int found =
            viewModeCombo->findData(static_cast<int>(*startupViewModeValue));
        if (found >= 0) startupViewModeIndex = found;
    }

    auto resolutionLabel = new QLabel("Ray Resolution");
    addWidget(resolutionLabel);
    std::array<QDoubleSpinBox*, 3> resolutionSpins{nullptr, nullptr, nullptr};
    const std::array<const char*, 3> axisNames{"X", "Y", "Z"};
    for (std::size_t i = 0; i < 3; ++i)
    {
        auto spin = new QDoubleSpinBox();
        useLightSpinArrows(spin);
        spin->setDecimals(6);
        spin->setRange(1.0e-6, 1.0e9);
        spin->setValue(0.000625);
        spin->setPrefix(QString("%1 ").arg(axisNames[i]));
        addWidget(spin);
        resolutionSpins[i] = spin;
    }

    auto applyResolutionButton = new QPushButton("Apply");
    addWidget(applyResolutionButton);
    if (!debugUi)
    {
        resolutionLabel->hide();
        for (auto* spin : resolutionSpins) spin->hide();
        applyResolutionButton->hide();
    }

    auto continuousCheck = new QCheckBox("Continuous update");
    continuousCheck->setChecked(true);
    addWidget(continuousCheck);

    addDivider();

    auto designTitle = new QLabel("Design:");
    designTitle->setFont(titleFont);
    addWidget(designTitle);

    auto shellButton = new QPushButton("Shell");
    addWidget(shellButton);

    auto cancelShellButton = new QPushButton("Cancel shell");
    cancelShellButton->setEnabled(false);
    addWidget(cancelShellButton);

    addDivider();

    auto toolTitle = new QLabel("Tool:");
    toolTitle->setFont(titleFont);
    addWidget(toolTitle);

    auto toolCombo = new QComboBox();
    toolCombo->addItem("None", static_cast<int>(app::ToolType::None));
    toolCombo->addItem("Bull nose", static_cast<int>(app::ToolType::BullNose));
    toolCombo->addItem("Flat nose", static_cast<int>(app::ToolType::FlatNose));
    toolCombo->addItem("Ball nose", static_cast<int>(app::ToolType::BallNose));
    toolCombo->addItem("Sphere", static_cast<int>(app::ToolType::Sphere));
    toolCombo->addItem("Grinding wheel", static_cast<int>(app::ToolType::GrindingWheel));
    toolCombo->addItem("Tool library", -1);
    addWidget(toolCombo);

    addWidget(new QLabel("Radius:"));
    auto radiusSpin = new QDoubleSpinBox();
    useLightSpinArrows(radiusSpin);
    radiusSpin->setDecimals(6);
    radiusSpin->setRange(1.0e-9, 1.0e9);
    radiusSpin->setValue(app::Parameter::instance().toolRadius());
    addWidget(radiusSpin);

    addWidget(new QLabel("Length:"));
    auto lengthSpin = new QDoubleSpinBox();
    useLightSpinArrows(lengthSpin);
    lengthSpin->setDecimals(6);
    lengthSpin->setRange(1.0e-9, 1.0e9);
    lengthSpin->setValue(app::Parameter::instance().toolLength());
    addWidget(lengthSpin);

    auto sweptVolumeCheck = new QCheckBox("Show swept volume");
    sweptVolumeCheck->setChecked(app::Parameter::instance().sweptVolume());
    addWidget(sweptVolumeCheck);

    auto lastSweptOnlyCheck = new QCheckBox("Show last swept volume only");
    lastSweptOnlyCheck->setChecked(app::Parameter::instance().showLastSweptVolumeOnly());
    addWidget(lastSweptOnlyCheck);

    auto clearSweptButton = new QPushButton("Clear swept volume");
    addWidget(clearSweptButton);

    addDivider();

    auto operationTitle = new QLabel("Operation:");
    operationTitle->setFont(titleFont);
    addWidget(operationTitle);

    auto booleanCombo = new QComboBox();
    const QString noneKeys =
        QKeySequence(Qt::CTRL | Qt::Key_N).toString(QKeySequence::NativeText);
    const QString probeKeys =
        QKeySequence(Qt::CTRL | Qt::Key_P).toString(QKeySequence::NativeText);
    const QString subtractKeys =
        QKeySequence(Qt::CTRL | Qt::Key_S).toString(QKeySequence::NativeText);
    const QString unionKeys =
        QKeySequence(Qt::CTRL | Qt::Key_U).toString(QKeySequence::NativeText);
    const QString inspectKeys =
        QKeySequence(Qt::CTRL | Qt::Key_I).toString(QKeySequence::NativeText);
    booleanCombo->addItem(QString("None (%1)").arg(noneKeys),
                          static_cast<int>(app::BooleanOp::None));
    booleanCombo->addItem(QString("Probe (%1)").arg(probeKeys),
                          static_cast<int>(app::BooleanOp::Probe));
    booleanCombo->addItem(QString("Subtraction (%1)").arg(subtractKeys),
                          static_cast<int>(app::BooleanOp::Subtraction));
    booleanCombo->addItem(QString("Union (%1)").arg(unionKeys),
                          static_cast<int>(app::BooleanOp::Union));
    booleanCombo->addItem(QString("Inspection (%1)").arg(inspectKeys),
                          static_cast<int>(app::BooleanOp::Inspection));
    booleanCombo->setCurrentIndex(
        booleanCombo->findData(static_cast<int>(app::BooleanOp::None)));
    booleanCombo->setToolTip(
        QString("None: %1 (exit data collection)\nProbe: %2 (record poses, no boolean)\n"
                "Subtraction: %3\nUnion: %4\nInspection: %5")
            .arg(noneKeys, probeKeys, subtractKeys, unionKeys, inspectKeys));
    addWidget(booleanCombo);

    addDivider();
    column->addSpacing(20);

    auto quitButton = new QPushButton("Quit");
    addWidget(quitButton);

    column->addStretch(1);

    auto panelScroll = new QScrollArea();
    panelScroll->setObjectName("sidePanel");
    panelScroll->setAutoFillBackground(true);
    panelScroll->setWidget(panel);
    panelScroll->setWidgetResizable(true);
    panelScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    panelScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    panelScroll->setFrameShape(QFrame::NoFrame);
    panelScroll->setMinimumWidth(236);

    // --- Ray resolution helpers --------------------------------------------
    auto readResolution = [resolutionSpins]() {
        return app::Point3d{resolutionSpins[0]->value(),
                            resolutionSpins[1]->value(),
                            resolutionSpins[2]->value()};
    };

    auto writeResolution = [resolutionSpins](const app::Point3d& resolution) {
        for (std::size_t i = 0; i < 3; ++i) resolutionSpins[i]->setValue(resolution[i]);
    };

    // Editing any of the three fields commits the whole resolution to the
    // store, so whatever casts rays next reads the current value from there.
    for (auto* spin : resolutionSpins)
    {
        QObject::connect(spin, &QDoubleSpinBox::valueChanged, [readResolution](double) {
            app::Parameter::instance().setRayResolution(readResolution());
        });
    }

    // A starting resolution that suits the model: rayDivisions steps across the
    // bounding box, so the grid density is independent of the model's units.
    auto defaultResolution = [](const app::BRep& brep) {
        const int divisions = app::Parameter::instance().rayDivisions();

        const app::BoundingBox bounds = app::BoundingBox::fromBRep(brep);
        const double fallback = std::max(bounds.diagonal(), 1.0) / divisions;

        app::Point3d resolution{fallback, fallback, fallback};
        for (std::size_t i = 0; i < 3; ++i)
        {
            const double extent = bounds.extent(i);
            if (extent > 0.0) resolution[i] = extent / divisions;
        }
        return resolution;
    };

    // Default cutter radius: 5% of the BRep bounding-box diagonal, in model space.
    // Default length matches the previous fixed shank factor (2.8 × radius).
    constexpr double defaultToolLengthFactor = 2.8;
    auto defaultToolRadius = [](const app::BRep& brep) {
        const app::BoundingBox bounds = app::BoundingBox::fromBRep(brep);
        const double diagonal = bounds.valid() ? bounds.diagonal() : 1.0;
        return 0.05 * std::max(diagonal, 1.0e-9);
    };

    auto writeToolRadius = [radiusSpin](double radius) {
        radiusSpin->blockSignals(true);
        radiusSpin->setValue(radius);
        radiusSpin->blockSignals(false);
    };

    auto writeToolLength = [lengthSpin](double length) {
        lengthSpin->blockSignals(true);
        lengthSpin->setValue(length);
        lengthSpin->blockSignals(false);
    };

    auto seedToolSize = [&](const app::BRep& brep) {
        const double radius = defaultToolRadius(brep);
        const double length = radius * defaultToolLengthFactor;
        app::Parameter::instance().setToolRadius(radius);
        app::Parameter::instance().setToolLength(length);
        writeToolRadius(radius);
        writeToolLength(length);
    };

    // --- Dockable left Controls and right Simulation, Vulkan in the center --
    auto simPanel = new app::SimulationPanel();
    simPanel->setObjectName("simPanel");
    simPanel->setAutoFillBackground(true);

    const auto dockFeatures = QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable
                            | QDockWidget::DockWidgetClosable;

    auto controlsDock = new QDockWidget("Controls", mainWindow);
    controlsDock->setObjectName("controlsDock");
    controlsDock->setWidget(panelScroll);
    controlsDock->setAllowedAreas(Qt::AllDockWidgetAreas);
    controlsDock->setFeatures(dockFeatures);
    controlsDock->setMinimumWidth(236);

    auto simDock = new QDockWidget("Simulation", mainWindow);
    simDock->setObjectName("simDock");
    simDock->setWidget(simPanel);
    simDock->setAllowedAreas(Qt::AllDockWidgetAreas);
    simDock->setFeatures(dockFeatures);
    simDock->setMinimumWidth(300);

    mainWindow->setDockNestingEnabled(true);
    mainWindow->addDockWidget(Qt::LeftDockWidgetArea, controlsDock);
    mainWindow->addDockWidget(Qt::RightDockWidgetArea, simDock);
    simDock->hide();

    auto* viewMenu = mainWindow->menuBar()->addMenu("View");
    viewMenu->addAction(controlsDock->toggleViewAction());
    viewMenu->addAction(simDock->toggleViewAction());

    renderWidget->setObjectName("central");
    renderWidget->setAutoFillBackground(true);
    mainWindow->setCentralWidget(renderWidget);
    mainWindow->setGeometry(windowTraits->x, windowTraits->y,
                            windowTraits->width + 236, windowTraits->height);
    mainWindow->show();
    application.processEvents();

    // Bridges BRep geometry into the VSG scene. Populated before the viewer is
    // initialized so the camera is framed around real geometry.
    auto renderManager = std::make_shared<app::RenderManager>(viewer, vsg_scene, options);
    simPanel->setRenderManager(renderManager);
    renderManager->setProfilingEnabled(profileCuts);
    if (profileCuts)
    {
        std::cout << "profiling enabled: one line per boolean cut\n";
        std::cout.flush();
    }

    const app::BRep startupBRep = createCylinderBRep();
    seedToolSize(startupBRep);
    renderManager->showBRep(startupBRep);
    writeResolution(defaultResolution(startupBRep));

    vsg::ref_ptr<vsg::Camera> camera;
    vsg::ref_ptr<vsg::LookAt> initialLookAt;
    vsg::ref_ptr<app::ControlCube> controlCube;
    auto trackball = initializeViewer(window, viewer, windowTraits, vsg_scene, options,
                                      camera, initialLookAt, controlCube);
    renderManager->configureRayBudgets(windowTraits->device);

    // Wire up the panel controls.
    QObject::connect(quitButton, &QPushButton::clicked, &application, &QApplication::quit);
    QObject::connect(continuousCheck, &QCheckBox::toggled, [viewer](bool on) {
        app::Parameter::instance().setContinuousUpdate(on);
        viewer->continuousUpdate = app::Parameter::instance().continuousUpdate();
    });
    QObject::connect(shellButton, &QPushButton::clicked,
                     [mainWindow, renderManager, radiusSpin, cancelShellButton]() {
                         bool ok = false;
                         const double fallback =
                             radiusSpin->value() > 0.0 ? radiusSpin->value() * 0.25 : 0.01;
                         const double thickness = QInputDialog::getDouble(
                             mainWindow, "Shell", "Thickness:", fallback, 1.0e-9, 1.0e9, 6, &ok);
                         if (!ok) return;
                         if (!renderManager->shellStock(thickness))
                         {
                             QMessageBox::warning(
                                 mainWindow, "Shell",
                                 "No simulation stock to shell, or thickness is invalid.");
                             return;
                         }
                         cancelShellButton->setEnabled(renderManager->hasPreShellCache());
                     });
    QObject::connect(cancelShellButton, &QPushButton::clicked,
                     [mainWindow, renderManager, cancelShellButton]() {
                         if (!renderManager->cancelShell())
                         {
                             QMessageBox::information(mainWindow, "Cancel shell",
                                                      "Nothing to cancel.");
                             cancelShellButton->setEnabled(false);
                             return;
                         }
                         cancelShellButton->setEnabled(renderManager->hasPreShellCache());
                     });
    QObject::connect(resetButton, &QPushButton::clicked,
                     [trackball, initialLookAt]() {
                         if (trackball && initialLookAt)
                             trackball->setViewpoint(vsg::LookAt::create(*initialLookAt), 1.0);
                     });
    app::ToolManagerDialog* toolManager = nullptr;
    const auto ensureToolManager = [mainWindow, windowTraits, options, interval, simPanel,
                                    &toolManager]() -> app::ToolManagerDialog* {
        if (!toolManager)
        {
            toolManager = new app::ToolManagerDialog(windowTraits, options, interval, mainWindow);
            QObject::connect(toolManager, &app::ToolManagerDialog::toolLibraryEntryChanged,
                             simPanel, &app::SimulationPanel::updateLibraryEntry);
        }
        return toolManager;
    };
    QObject::connect(toolCombo, &QComboBox::currentIndexChanged,
                     [renderManager, toolCombo, simPanel, ensureToolManager](int index) {
                         if (toolCombo->itemData(index).toInt() == -1)
                         {
                             toolCombo->blockSignals(true);
                             const int current =
                                 static_cast<int>(app::Parameter::instance().toolType());
                             const int restore = toolCombo->findData(current);
                             if (restore >= 0) toolCombo->setCurrentIndex(restore);
                             toolCombo->blockSignals(false);

                             auto* manager = ensureToolManager();
                             manager->show();
                             manager->raise();
                             manager->activateWindow();
                             return;
                         }
                         const auto type =
                             static_cast<app::ToolType>(toolCombo->itemData(index).toInt());
                         app::Parameter::instance().setToolType(type);
                         app::Parameter::instance().setToolShankRadius(0.0);
                         app::Parameter::instance().setToolShankLength(0.0);
                         renderManager->setToolType(type);
                         renderManager->updateToolGeometry();
                         simPanel->clearLibrarySelection();
                     });
    QObject::connect(simPanel, &app::SimulationPanel::toolLibraryApplied,
                     [toolCombo, radiusSpin, lengthSpin](int, double radius,
                                                          double cuttingLength, double,
                                                          double) {
                         toolCombo->blockSignals(true);
                         const int libraryIndex = toolCombo->findData(-1);
                         if (libraryIndex >= 0) toolCombo->setCurrentIndex(libraryIndex);
                         toolCombo->blockSignals(false);
                         radiusSpin->blockSignals(true);
                         radiusSpin->setValue(radius);
                         radiusSpin->blockSignals(false);
                         lengthSpin->blockSignals(true);
                         lengthSpin->setValue(cuttingLength);
                         lengthSpin->blockSignals(false);
                     });
    QObject::connect(simPanel, &app::SimulationPanel::toolLibraryPreviewRequested,
                     [ensureToolManager](int toolType, double radius, double cuttingLength,
                                         double shankLength, double shankRadius) {
                         auto* manager = ensureToolManager();
                         manager->previewTool(static_cast<app::ToolType>(toolType), radius,
                                              cuttingLength, shankLength, shankRadius);
                         manager->show();
                         manager->raise();
                         manager->activateWindow();
                     });
    QObject::connect(radiusSpin, &QDoubleSpinBox::valueChanged,
                     [renderManager](double radius) {
                         app::Parameter::instance().setToolRadius(radius);
                         renderManager->updateToolGeometry();
                     });
    QObject::connect(lengthSpin, &QDoubleSpinBox::valueChanged,
                     [renderManager](double length) {
                         app::Parameter::instance().setToolLength(length);
                         renderManager->updateToolGeometry();
                     });
    QObject::connect(sweptVolumeCheck, &QCheckBox::toggled, [renderManager](bool on) {
        app::Parameter::instance().setSweptVolume(on);
        renderManager->setSweptVolumeVisible(on);
    });
    QObject::connect(lastSweptOnlyCheck, &QCheckBox::toggled, [](bool on) {
        app::Parameter::instance().setShowLastSweptVolumeOnly(on);
    });
    QObject::connect(clearSweptButton, &QPushButton::clicked, [renderManager]() {
        renderManager->clearSweptVolume();
    });
    QObject::connect(booleanCombo, &QComboBox::currentIndexChanged,
                     [renderManager, booleanCombo, simPanel](int index) {
                         const auto op =
                             static_cast<app::BooleanOp>(booleanCombo->itemData(index).toInt());
                         renderManager->setBooleanOp(op);
                         simPanel->notifyBooleanOp(op);
                     });

    auto selectBooleanOp = [booleanCombo](app::BooleanOp op) {
        const int want = static_cast<int>(op);
        for (int i = 0; i < booleanCombo->count(); ++i)
        {
            if (booleanCombo->itemData(i).toInt() == want)
            {
                booleanCombo->setCurrentIndex(i);
                return;
            }
        }
    };

    auto noneShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_N), mainWindow);
    noneShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(noneShortcut, &QShortcut::activated,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::None); });

    auto probeShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_P), mainWindow);
    probeShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(probeShortcut, &QShortcut::activated,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Probe); });

    auto subtractShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_S), mainWindow);
    subtractShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(subtractShortcut, &QShortcut::activated,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Subtraction); });

    auto unionShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_U), mainWindow);
    unionShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(unionShortcut, &QShortcut::activated,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Union); });

    auto inspectShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_I), mainWindow);
    inspectShortcut->setContext(Qt::ApplicationShortcut);
    QObject::connect(inspectShortcut, &QShortcut::activated,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Inspection); });
    QObject::connect(simPanel, &app::SimulationPanel::noneOperationRequested,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::None); });
    QObject::connect(simPanel, &app::SimulationPanel::probeOperationRequested,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Probe); });
    QObject::connect(simPanel, &app::SimulationPanel::subtractionOperationRequested,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Subtraction); });
    QObject::connect(simPanel, &app::SimulationPanel::unionOperationRequested,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Union); });
    QObject::connect(simPanel, &app::SimulationPanel::inspectionOperationRequested,
                     [selectBooleanOp]() { selectBooleanOp(app::BooleanOp::Inspection); });
    booleanCombo->setContextMenuPolicy(Qt::CustomContextMenu);
    QObject::connect(booleanCombo, &QWidget::customContextMenuRequested,
                     [booleanCombo, simPanel](const QPoint& pos) {
                         simPanel->popupExitCollectionMenu(booleanCombo->mapToGlobal(pos));
                     });
    window->installEventFilter(new ViewportContextMenuFilter(simPanel, controlCube, window));
    // Cast rays through whatever BRep is on display, at the resolution held in
    // the store, and hand the result to the RenderManager. Resolutions already
    // cast through this BRep are still held there, so switching between view
    // modes only pays for the casting once.
    auto buildRayModel = [renderManager]() {
        const auto& brep = renderManager->currentBRep();
        if (!brep) return;

        // The resolution is cast exactly as it was typed. Being too fine to
        // draw is never a reason to change it: the renderer thins the model out
        // by grid layers to whatever the GPU can take, while the model it
        // thinned keeps every ray. Only a resolution too fine to hold in memory
        // is refused, and then the message says what would fit rather than
        // quietly substituting it.
        const app::Point3d resolution = app::Parameter::instance().rayResolution();

        if (renderManager->useCachedRayModel(resolution)) return;

        renderManager->setRayModel(app::RayModel::fromBRep(*brep, resolution));
    };

    const int simulationModeIndex =
        viewModeCombo->findData(static_cast<int>(app::ViewMode::RayGS));

    QObject::connect(viewModeCombo, &QComboBox::currentIndexChanged,
                     [mainWindow, renderManager, viewModeCombo, buildRayModel, simDock](int index) {
                         const auto mode =
                             static_cast<app::ViewMode>(viewModeCombo->itemData(index).toInt());

                         app::Parameter& store = app::Parameter::instance();
                         store.setViewMode(mode);
                         simDock->setVisible(app::usesRayModel(store.viewMode()));

                         try
                         {
                             // Cast first so switching to Ray draws the rays
                             // rather than briefly falling back to the surface.
                             if (app::usesRayModel(store.viewMode())) buildRayModel();
                             renderManager->setViewMode(store.viewMode());
                         }
                         catch (const std::exception& e)
                         {
                             QMessageBox::warning(mainWindow, "View mode", e.what());
                         }
                     });

    // Apply recasts the rays at the resolution now in the spin boxes. If the
    // ray view is not already showing, switching to it does the recast, so the
    // work is never done twice.
    QObject::connect(applyResolutionButton, &QPushButton::clicked,
                     [mainWindow, viewModeCombo, simulationModeIndex, buildRayModel]() {
                         try
                         {
                             // Already on Ray or Ray-GS: recast in place and
                             // stay in whichever of the two is showing.
                             if (app::usesRayModel(app::Parameter::instance().viewMode()))
                                 buildRayModel();
                             else
                                 viewModeCombo->setCurrentIndex(simulationModeIndex);
                         }
                         catch (const std::exception& e)
                         {
                             QMessageBox::warning(mainWindow, "Ray resolution", e.what());
                         }
                     });

    // Import: read a model file into a BRep, validate watertightness, then hand
    // it to the RenderManager. Shared by both import buttons and the optional
    // --import startup option.
    auto importModel = [mainWindow, renderManager, buildRayModel,
                        writeResolution, defaultResolution,
                        seedToolSize](const std::string& path) {
        try
        {
            const app::TriangleMesh mesh = importMesh(path);

            const app::BRep brep = app::BRep::fromTriangles(mesh);
            const app::BRep::ValidationResult v = brep.validate();

            if (!v.watertight)
            {
                QMessageBox::warning(
                    mainWindow, "Mesh is not watertight",
                    QString("The imported mesh is not watertight.\n\n"
                            "Triangles: %1\nUnique vertices: %2\n"
                            "Boundary edges: %3\nNon-manifold edges: %4\n"
                            "Degenerate faces: %5")
                        .arg(static_cast<qulonglong>(mesh.triangles.size()))
                        .arg(static_cast<qulonglong>(brep.vertexCount()))
                        .arg(static_cast<qulonglong>(v.boundaryEdgeCount))
                        .arg(static_cast<qulonglong>(v.nonManifoldEdgeCount))
                        .arg(static_cast<qulonglong>(v.degenerateFaceCount)));
            }

            app::Parameter::instance().setLastImportPath(path);

            // Reseed radius/length for the new model before showBRep so the tool
            // mesh (if any) is rebuilt at the default size.
            seedToolSize(brep);

            renderManager->showBRep(brep);

            // Reseed the resolution for the new model's size, then recast if
            // the ray view is the one being shown (showBRep drops the rays that
            // belonged to the previous model). Writing the fields commits the
            // new resolution to the store.
            writeResolution(defaultResolution(brep));
            if (app::usesRayModel(app::Parameter::instance().viewMode())) buildRayModel();
        }
        catch (const std::exception& e)
        {
            QMessageBox::critical(mainWindow, "Import failed", e.what());
        }
    };

    // Both buttons open the same dialog, differing only in which format it
    // offers first; the reader is still picked from the chosen file's extension.
    auto openImportDialog = [mainWindow, importModel](const QString& title,
                                                      const QString& filters) {
        // Reopen where the last import came from.
        const QString startDir =
            QString::fromStdString(app::Parameter::instance().lastImportPath());

        const QString fileName =
            QFileDialog::getOpenFileName(mainWindow, title, startDir, filters);
        if (!fileName.isEmpty()) importModel(fileName.toStdString());
    };

    QObject::connect(importStlButton, &QPushButton::clicked, [openImportDialog]() {
        openImportDialog("Import STL file",
                         "STL files (*.stl);;3MF files (*.3mf);;All files (*)");
    });

    QObject::connect(import3mfButton, &QPushButton::clicked, [openImportDialog]() {
        openImportDialog("Import 3MF file",
                         "3MF files (*.3mf);;STL files (*.stl);;All files (*)");
    });

    if (interval >= 0) viewer->setInterval(interval);
    viewer->continuousUpdate = parameters.continuousUpdate();
    continuousCheck->setChecked(parameters.continuousUpdate());

    // Tool tracking has to see moves before (or at least as well as) the
    // trackball; it never marks events handled, so orbit/zoom keep working.
    viewer->addEventHandler(app::ToolTracker::create(renderManager, camera, simPanel, controlCube));
    renderManager->setCamera(camera);
    viewer->addEventHandler(renderManager->createCameraSettleHandler());
    viewer->addEventHandler(vsg::CloseHandler::create(viewer));
    viewer->compile();

    // Optionally import a model immediately at startup.
    if (!startupModel.empty()) importModel(startupModel);

    // Applying the startup view mode through the combo box keeps the UI and the
    // RenderManager in step.
    viewModeCombo->setCurrentIndex(startupViewModeIndex);
    toolCombo->setCurrentIndex(startupToolIndex);

    return application.exec();
}
catch (const vsg::Exception& e)
{
    std::cerr << "vsg::Exception: " << e.message << " (VkResult " << e.result << ")" << std::endl;
    return 1;
}
catch (const std::exception& e)
{
    std::cerr << "std::exception: " << e.what() << std::endl;
    return 1;
}
