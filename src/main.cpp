// vsg_qt_cube
//
// Creates a Qt main window (QMainWindow) with a left-hand control panel
// (vertical QGridLayout) beside a VulkanSceneGraph rendering surface embedded
// via vsgQt. Geometry is held as a BRep and drawn by the RenderManager: the
// scene starts with a unit cube, and STL or 3MF files can be imported at
// runtime via the panel's import buttons. The panel's view mode selector switches
// between facet and wireframe rendering of that BRep.
//
// Controls: left-drag to rotate, right-drag / wheel to zoom, middle-drag to pan.

#include <vsg/all.h>

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QWidget>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QMessageBox>
#include <QtGui/QFont>
#include <QtGui/QKeySequence>
#include <QtGui/QShortcut>
#include <QtGui/QStandardItemModel>

#include <vsgQt/Window.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <iostream>
#include <memory>

#include "BRep.h"
#include "BoundingBox.h"
#include "Parameter.h"
#include "RayModel.h"
#include "RenderManager.h"
#include "StlImporter.h"
#include "ThreeMfImporter.h"
#include "ToolTracker.h"
#include "ToolType.h"

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
// to the view mode selector.
app::BRep createCubeBRep()
{
    const vsg::vec3 corners[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};

    // Two counter-clockwise triangles per face, outward facing.
    const int triangles[12][3] = {
        {0, 3, 2}, {0, 2, 1}, // -Z
        {4, 5, 6}, {4, 6, 7}, // +Z
        {0, 1, 5}, {0, 5, 4}, // -Y
        {1, 2, 6}, {1, 6, 5}, // +X
        {2, 3, 7}, {2, 7, 6}, // +Y
        {3, 0, 4}, {3, 4, 7}  // -X
    };

    app::TriangleMesh mesh;
    mesh.name = "cube";
    mesh.triangles.reserve(12);

    for (const auto& tri : triangles)
    {
        app::MeshTriangle t;
        t.v0 = corners[tri[0]];
        t.v1 = corners[tri[1]];
        t.v2 = corners[tri[2]];
        t.normal = vsg::normalize(vsg::cross(t.v1 - t.v0, t.v2 - t.v0));
        mesh.triangles.push_back(t);
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
                                              vsg::ref_ptr<vsg::Camera>& out_camera,
                                              vsg::ref_ptr<vsg::LookAt>& out_lookAt)
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
    trackball->addWindow(*window);
    viewer->addEventHandler(trackball);

    auto commandGraph = vsg::createCommandGraphForView(*window, camera, vsg_scene);
    viewer->addRecordAndSubmitTaskAndPresentation({commandGraph});

    // Snapshot the starting viewpoint for the panel's "Reset View" action.
    out_camera = camera;
    out_lookAt = vsg::LookAt::create(*lookAt);

    return trackball;
}

} // namespace

int main(int argc, char* argv[])
try
{
    QApplication application(argc, argv);

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

    int startupViewModeIndex = 0;
    if (!startupViewMode.empty())
    {
        if (startupViewMode == "facet")
            startupViewModeIndex = 0;
        else if (startupViewMode == "wireframe")
            startupViewModeIndex = 1;
        else if (startupViewMode == "ray")
            startupViewModeIndex = 2;
        else if (startupViewMode == "ray-gs")
            startupViewModeIndex = 3;
        else
        {
            std::cerr << "unknown --view-mode '" << startupViewMode
                      << "', expected facet, wireframe, ray or ray-gs" << std::endl;
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

    // --- Left-hand control panel with a vertical grid layout ---------------
    auto panel = new QWidget();
    panel->setFixedWidth(220);

    auto grid = new QGridLayout(panel);
    grid->setContentsMargins(8, 8, 8, 8);
    grid->setSpacing(6);
    grid->setAlignment(Qt::AlignTop);

    int row = 0;

    // Import button at the very top of the panel.
    auto importStlButton = new QPushButton("Import STL…");
    grid->addWidget(importStlButton, row++, 0);

    auto import3mfButton = new QPushButton("Import 3MF…");
    grid->addWidget(import3mfButton, row++, 0);

    auto title = new QLabel("Scene Controls");
    QFont titleFont = title->font();
    titleFont.setBold(true);
    title->setFont(titleFont);
    grid->addWidget(title, row++, 0);

    auto resetButton = new QPushButton("Reset View");
    grid->addWidget(resetButton, row++, 0);

    grid->addWidget(new QLabel("View Mode"), row++, 0);

    auto viewModeCombo = new QComboBox();
    viewModeCombo->addItem("Facet", static_cast<int>(app::ViewMode::Facet));
    viewModeCombo->addItem("Wireframe", static_cast<int>(app::ViewMode::Wireframe));
    viewModeCombo->addItem("Ray", static_cast<int>(app::ViewMode::Ray));
    viewModeCombo->addItem("Simulation", static_cast<int>(app::ViewMode::RayGS));
    grid->addWidget(viewModeCombo, row++, 0);

    // Ray tessellation resolution: the spacing of the cast grid along each
    // axis. Seeded from the model's size; edit and press Apply to recast.
    grid->addWidget(new QLabel("Ray Resolution"), row++, 0);

    auto resolutionPanel = new QWidget();
    auto resolutionGrid = new QGridLayout(resolutionPanel);
    resolutionGrid->setContentsMargins(0, 0, 0, 0);
    resolutionGrid->setSpacing(4);

    std::array<QDoubleSpinBox*, 3> resolutionSpins{nullptr, nullptr, nullptr};
    const std::array<const char*, 3> axisNames{"X", "Y", "Z"};
    for (std::size_t i = 0; i < 3; ++i)
    {
        auto spin = new QDoubleSpinBox();
        spin->setDecimals(6);
        spin->setRange(1.0e-6, 1.0e9);
        spin->setValue(0.000625);

        const int gridRow = static_cast<int>(i);
        resolutionGrid->addWidget(new QLabel(axisNames[i]), gridRow, 0);
        resolutionGrid->addWidget(spin, gridRow, 1);
        resolutionSpins[i] = spin;
    }
    resolutionGrid->setColumnStretch(1, 1);
    grid->addWidget(resolutionPanel, row++, 0);

    auto applyResolutionButton = new QPushButton("Apply");
    grid->addWidget(applyResolutionButton, row++, 0);

    auto continuousCheck = new QCheckBox("Continuous update");
    continuousCheck->setChecked(true);
    grid->addWidget(continuousCheck, row++, 0);

    // Horizontal rules between panel sections. Default HLine is ~2px; five
    // times that keeps the break readable in the narrow sidebar.
    auto makeDivider = []() {
        auto* line = new QFrame();
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        line->setFixedHeight(10);
        line->setStyleSheet("QFrame { color: #9a9a9a; margin-top: 4px; margin-bottom: 4px; }");
        return line;
    };

    grid->addWidget(makeDivider(), row++, 0);

    grid->addWidget(new QLabel("Tool"), row++, 0);

    auto toolCombo = new QComboBox();
    toolCombo->addItem("None", static_cast<int>(app::ToolType::None));
    toolCombo->addItem("Bull nose", static_cast<int>(app::ToolType::BullNose));
    toolCombo->addItem("Flat nose", static_cast<int>(app::ToolType::FlatNose));
    toolCombo->addItem("Ball nose", static_cast<int>(app::ToolType::BallNose));
    toolCombo->addItem("Sphere", static_cast<int>(app::ToolType::Sphere));
    grid->addWidget(toolCombo, row++, 0);

    grid->addWidget(new QLabel("Radius"), row++, 0);

    auto radiusSpin = new QDoubleSpinBox();
    radiusSpin->setDecimals(6);
    radiusSpin->setRange(1.0e-9, 1.0e9);
    radiusSpin->setValue(app::Parameter::instance().toolRadius());
    grid->addWidget(radiusSpin, row++, 0);

    grid->addWidget(new QLabel("Length"), row++, 0);

    auto lengthSpin = new QDoubleSpinBox();
    lengthSpin->setDecimals(6);
    lengthSpin->setRange(1.0e-9, 1.0e9);
    lengthSpin->setValue(app::Parameter::instance().toolLength());
    grid->addWidget(lengthSpin, row++, 0);

    auto sweptVolumeCheck = new QCheckBox("Show swept volume");
    sweptVolumeCheck->setChecked(app::Parameter::instance().sweptVolume());
    grid->addWidget(sweptVolumeCheck, row++, 0);

    auto lastSweptOnlyCheck = new QCheckBox("Show last swept volume only");
    lastSweptOnlyCheck->setChecked(app::Parameter::instance().showLastSweptVolumeOnly());
    grid->addWidget(lastSweptOnlyCheck, row++, 0);

    auto clearSweptButton = new QPushButton("Clear swept volume");
    grid->addWidget(clearSweptButton, row++, 0);

    grid->addWidget(makeDivider(), row++, 0);

    grid->addWidget(new QLabel("Boolean operation"), row++, 0);

    auto booleanCombo = new QComboBox();
    const QString noneKeys =
        QKeySequence(Qt::CTRL | Qt::Key_N).toString(QKeySequence::NativeText);
    const QString subtractKeys =
        QKeySequence(Qt::CTRL | Qt::Key_S).toString(QKeySequence::NativeText);
    const QString unionKeys =
        QKeySequence(Qt::CTRL | Qt::Key_U).toString(QKeySequence::NativeText);
    const QString inspectKeys =
        QKeySequence(Qt::CTRL | Qt::Key_I).toString(QKeySequence::NativeText);
    booleanCombo->addItem(QString("None (%1)").arg(noneKeys),
                          static_cast<int>(app::BooleanOp::None));
    booleanCombo->addItem(QString("Subtraction (%1)").arg(subtractKeys),
                          static_cast<int>(app::BooleanOp::Subtraction));
    booleanCombo->addItem(QString("Union (%1)").arg(unionKeys),
                          static_cast<int>(app::BooleanOp::Union));
    booleanCombo->addItem(QString("Inspection (%1)").arg(inspectKeys),
                          static_cast<int>(app::BooleanOp::Inspection));
    booleanCombo->setCurrentIndex(0);
    booleanCombo->setToolTip(
        QString("None: %1\nSubtraction: %2\nUnion: %3\nInspection: %4")
            .arg(noneKeys, subtractKeys, unionKeys, inspectKeys));
    grid->addWidget(booleanCombo, row++, 0);

    auto quitButton = new QPushButton("Quit");
    grid->addWidget(quitButton, row++, 0);

    // Push the controls to the top by absorbing the remaining vertical space.
    grid->setRowStretch(row, 1);

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

    // --- Compose panel (left) + render surface (right) ---------------------
    auto central = new QWidget();
    auto hbox = new QHBoxLayout(central);
    hbox->setContentsMargins(0, 0, 0, 0);
    hbox->setSpacing(0);
    hbox->addWidget(panel);
    hbox->addWidget(renderWidget, 1);

    mainWindow->setCentralWidget(central);
    mainWindow->setGeometry(windowTraits->x, windowTraits->y,
                            panel->width() + windowTraits->width, windowTraits->height);
    mainWindow->show();
    application.processEvents();

    // Bridges BRep geometry into the VSG scene. Populated before the viewer is
    // initialized so the camera is framed around real geometry.
    auto renderManager = std::make_shared<app::RenderManager>(viewer, vsg_scene, options);
    renderManager->setProfilingEnabled(profileCuts);
    if (profileCuts)
    {
        std::cout << "profiling enabled: one line per boolean cut\n";
        std::cout.flush();
    }

    const app::BRep startupBRep = createCubeBRep();
    seedToolSize(startupBRep);
    renderManager->showBRep(startupBRep);
    writeResolution(defaultResolution(startupBRep));

    vsg::ref_ptr<vsg::Camera> camera;
    vsg::ref_ptr<vsg::LookAt> initialLookAt;
    auto trackball = initializeViewer(window, viewer, windowTraits, vsg_scene, camera, initialLookAt);

    // Wire up the panel controls.
    QObject::connect(quitButton, &QPushButton::clicked, &application, &QApplication::quit);
    QObject::connect(continuousCheck, &QCheckBox::toggled, [viewer](bool on) {
        app::Parameter::instance().setContinuousUpdate(on);
        viewer->continuousUpdate = app::Parameter::instance().continuousUpdate();
    });
    QObject::connect(resetButton, &QPushButton::clicked,
                     [trackball, initialLookAt]() {
                         if (trackball && initialLookAt)
                             trackball->setViewpoint(vsg::LookAt::create(*initialLookAt), 1.0);
                     });
    QObject::connect(toolCombo, &QComboBox::currentIndexChanged,
                     [renderManager, toolCombo](int index) {
                         const auto type =
                             static_cast<app::ToolType>(toolCombo->itemData(index).toInt());
                         app::Parameter::instance().setToolType(type);
                         renderManager->setToolType(type);
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
                     [renderManager, booleanCombo](int index) {
                         const auto op =
                             static_cast<app::BooleanOp>(booleanCombo->itemData(index).toInt());
                         renderManager->setBooleanOp(op);
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

    const int rayModeIndex = viewModeCombo->findData(static_cast<int>(app::ViewMode::Ray));

    QObject::connect(viewModeCombo, &QComboBox::currentIndexChanged,
                     [mainWindow, renderManager, viewModeCombo, buildRayModel](int index) {
                         const auto mode =
                             static_cast<app::ViewMode>(viewModeCombo->itemData(index).toInt());

                         app::Parameter& store = app::Parameter::instance();
                         store.setViewMode(mode);

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
                     [mainWindow, viewModeCombo, rayModeIndex, buildRayModel]() {
                         try
                         {
                             // Already on Ray or Ray-GS: recast in place and
                             // stay in whichever of the two is showing.
                             if (app::usesRayModel(app::Parameter::instance().viewMode()))
                                 buildRayModel();
                             else
                                 viewModeCombo->setCurrentIndex(rayModeIndex);
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
    viewer->addEventHandler(app::ToolTracker::create(renderManager, camera));
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
