#include "SimulationPanel.h"

#include <algorithm>
#include <cmath>

#include <QtCore/QEvent>
#include <QtCore/QTimer>
#include <QtGui/QContextMenuEvent>
#include <QtGui/QCursor>
#include <QtGui/QFont>
#include <QtGui/QHideEvent>
#include <QtGui/QStandardItemModel>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMenu>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QVBoxLayout>

#include "BooleanOp.h"
#include "BoundingBox.h"
#include "Parameter.h"
#include "RenderManager.h"
#include "SimulationMode.h"
#include "ToolType.h"
#include "ViewMode.h"

#include <vector>

namespace app
{
namespace
{

QTableWidgetItem* makeCell(double value, int decimals)
{
    auto* item = new QTableWidgetItem(QString::number(value, 'f', decimals));
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    return item;
}

QTableWidgetItem* makeEditableCell(double value, int decimals)
{
    auto* item = new QTableWidgetItem(QString::number(value, 'f', decimals));
    item->setFlags(item->flags() | Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    return item;
}

QTableWidgetItem* makeNullCell(bool label)
{
    auto* item = new QTableWidgetItem(label ? QStringLiteral("Null") : QString());
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignCenter);
    return item;
}

QTableWidgetItem* makeTextCell(const QString& text, Qt::Alignment align = Qt::AlignCenter)
{
    auto* item = new QTableWidgetItem(text);
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    item->setTextAlignment(align | Qt::AlignVCenter);
    return item;
}

bool parsePositive(const QString& text, double* out)
{
    bool ok = false;
    const double value = text.trimmed().toDouble(&ok);
    if (!ok || !(value > 0.0)) return false;
    *out = value;
    return true;
}

const char* toolTypeLabel(ToolType type)
{
    switch (type)
    {
    case ToolType::BullNose: return "Bull nose";
    case ToolType::FlatNose: return "Flat nose";
    case ToolType::BallNose: return "Ball nose";
    case ToolType::Sphere: return "Sphere";
    case ToolType::GrindingWheel: return "Grinding wheel";
    case ToolType::None: return "None";
    }
    return "None";
}

bool movedEnough(const ToolSample& a, const ToolSample& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > 1.0e-4) return true;

    const double da = a.a - b.a;
    const double db = a.b - b.b;
    const double dc = a.c - b.c;
    return (da * da + db * db + dc * dc) > (0.1 * 0.1);
}

} // namespace

SimulationPanel::SimulationPanel(QWidget* parent) : QWidget(parent)
{
    setMinimumWidth(300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 10, 10, 10);
    layout->setSpacing(10);

    auto* title = new QLabel("Simulation");
    QFont titleFont = title->font();
    titleFont.setBold(true);
    title->setFont(titleFont);
    layout->addWidget(title);

    _modeCombo = new QComboBox();
    _modeCombo->addItem("Interactive", static_cast<int>(SimulationMode::Interactive));
    _modeCombo->addItem("CL data", static_cast<int>(SimulationMode::ClData));
    _modeCombo->addItem("NC machining", static_cast<int>(SimulationMode::NcMachining));
    if (auto* model = qobject_cast<QStandardItemModel*>(_modeCombo->model()))
    {
        if (QStandardItem* nc = model->item(2)) nc->setEnabled(false);
    }
    layout->addWidget(_modeCombo);

    _clDataHost = new QWidget();
    auto* clRow = new QHBoxLayout(_clDataHost);
    clRow->setContentsMargins(0, 0, 0, 0);
    clRow->setSpacing(10);
    _helixButton = new QPushButton("Helix");
    _importAptButton = new QPushButton("Import APT");
    clRow->addWidget(_helixButton);
    clRow->addWidget(_importAptButton);
    layout->addWidget(_clDataHost);

    _tableHost = new QWidget();
    _tableHost->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* tableLayout = new QVBoxLayout(_tableHost);
    tableLayout->setContentsMargins(0, 0, 0, 0);
    tableLayout->setSpacing(10);

    auto setupTable = [](QTableWidget* table) {
        table->setSelectionBehavior(QAbstractItemView::SelectRows);
        table->setSelectionMode(QAbstractItemView::SingleSelection);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        table->verticalHeader()->setVisible(false);
        table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
        table->horizontalHeader()->setStretchLastSection(true);
        table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        table->setMinimumHeight(140);
    };

    auto* poseSection = new QWidget();
    poseSection->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* poseLayout = new QVBoxLayout(poseSection);
    poseLayout->setContentsMargins(0, 0, 0, 0);
    poseLayout->setSpacing(10);

    _table = new QTableWidget(0, 6);
    _table->setHorizontalHeaderLabels({"X", "Y", "Z", "A", "B", "C"});
    setupTable(_table);
    poseLayout->addWidget(_table, 1);

    auto* buttonRow = new QHBoxLayout();
    buttonRow->setContentsMargins(0, 0, 0, 0);
    buttonRow->setSpacing(10);
    _rerunButton = new QPushButton("Re run");
    _resetButton = new QPushButton("Reset");
    buttonRow->addWidget(_rerunButton);
    buttonRow->addWidget(_resetButton);
    poseLayout->addLayout(buttonRow);

    _waitSlider = new QSlider(Qt::Horizontal);
    _waitSlider->setRange(0, sliderMax);
    _waitSlider->setValue(0);
    _waitSlider->setToolTip("Rerun step: min = 1 row, max = 100 rows");
    poseLayout->addWidget(_waitSlider);

    auto* librarySection = new QWidget();
    librarySection->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    auto* libraryLayout = new QVBoxLayout(librarySection);
    libraryLayout->setContentsMargins(0, 0, 0, 0);
    libraryLayout->setSpacing(10);

    auto* divider = new QFrame();
    divider->setFrameShape(QFrame::HLine);
    divider->setFrameShadow(QFrame::Sunken);
    divider->setFixedHeight(2);
    divider->setStyleSheet("QFrame { color: #9a9a9a; }");
    libraryLayout->addWidget(divider);

    auto* libraryTitle = new QLabel("Tool library");
    libraryTitle->setFont(titleFont);
    libraryLayout->addWidget(libraryTitle);

    _toolTable = new QTableWidget(0, 6);
    _toolTable->setHorizontalHeaderLabels({"Tool ID", "Tool type", "Radius", "Cutting length",
                                           "Shank length", "Shank radius"});
    setupTable(_toolTable);
    // Pose table stays locked; library numbers can be edited (F2 / type / re-click).
    // Double-click still opens the Tool manager preview.
    _toolTable->setEditTriggers(QAbstractItemView::EditKeyPressed |
                                QAbstractItemView::AnyKeyPressed |
                                QAbstractItemView::SelectedClicked);
    _toolTable->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _toolTable->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    _toolTable->setWordWrap(false);
    _toolTable->horizontalHeader()->setStretchLastSection(false);
    _toolTable->horizontalHeader()->setMinimumSectionSize(72);
    const int toolColWidths[] = {72, 100, 90, 120, 110, 110};
    for (int col = 0; col < 6; ++col)
        _toolTable->setColumnWidth(col, toolColWidths[col]);
    libraryLayout->addWidget(_toolTable, 1);
    fillToolLibrary();

    auto* splitter = new QSplitter(Qt::Vertical);
    splitter->setChildrenCollapsible(false);
    splitter->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    splitter->addWidget(poseSection);
    splitter->addWidget(librarySection);
    splitter->setStretchFactor(0, 1);
    splitter->setStretchFactor(1, 1);
    tableLayout->addWidget(splitter, 1);

    layout->addWidget(_tableHost, 1);

    connect(_toolTable, &QTableWidget::itemSelectionChanged, this,
            &SimulationPanel::onToolLibrarySelectionChanged);
    connect(_toolTable, &QTableWidget::cellDoubleClicked, this,
            &SimulationPanel::onToolLibraryDoubleClicked);
    connect(_toolTable, &QTableWidget::itemChanged, this,
            &SimulationPanel::onToolLibraryCellChanged);

    _playTimer = new QTimer(this);
    _playTimer->setSingleShot(true);

    connect(_modeCombo, &QComboBox::currentIndexChanged, this, &SimulationPanel::onModeChanged);
    connect(_helixButton, &QPushButton::clicked, this, &SimulationPanel::onHelixClicked);
    connect(_rerunButton, &QPushButton::clicked, this, &SimulationPanel::onRerun);
    connect(_resetButton, &QPushButton::clicked, this, &SimulationPanel::onReset);
    connect(_waitSlider, &QSlider::valueChanged, this, &SimulationPanel::onWaitSliderChanged);
    connect(_playTimer, &QTimer::timeout, this, &SimulationPanel::onPlayStep);

    const auto children = findChildren<QWidget*>();
    for (QWidget* child : children)
    {
        child->setContextMenuPolicy(Qt::CustomContextMenu);
        child->installEventFilter(this);
    }
    setContextMenuPolicy(Qt::CustomContextMenu);
    installEventFilter(this);

    setTableVisibleForMode();
}

void SimulationPanel::setRenderManager(std::shared_ptr<RenderManager> renderManager)
{
    _renderManager = std::move(renderManager);
}

void SimulationPanel::record(const ToolPose& pose)
{
    if (_playing) return;

    const Parameter& store = Parameter::instance();
    if (store.simulationMode() != SimulationMode::Interactive) return;
    if (!usesRayModel(store.viewMode())) return;
    if (!recordsToolPath(store.booleanOp())) return;

    const ToolSample sample = toolSampleFromPose(pose);
    if (_lastRecorded && !movedEnough(*_lastRecorded, sample)) return;

    _lastRecorded = sample;
    _log.push(sample);
    if (_log.drainQueued) return;

    _log.drainQueued = true;
    QMetaObject::invokeMethod(this, &SimulationPanel::drainPoseBuffer, Qt::QueuedConnection);
}

bool SimulationPanel::eventFilter(QObject* watched, QEvent* event)
{
    if (event->type() == QEvent::ContextMenu)
    {
        const auto* context = static_cast<QContextMenuEvent*>(event);
        popupExitCollectionMenu(context->globalPos());
        return true;
    }
    return QWidget::eventFilter(watched, event);
}

void SimulationPanel::hideEvent(QHideEvent* event)
{
    stopPlayback();
    if (_pickingHelixAxis)
    {
        _reopenHelixDialog = false;
        endHelixAxisPick();
    }
    QWidget::hideEvent(event);
}

void SimulationPanel::notifyBooleanOp(BooleanOp op)
{
    if (recordsToolPath(op)) return;
    drainPoseBuffer();
    appendNullRow();
    _lastRecorded.reset();
}

void SimulationPanel::popupExitCollectionMenu(const QPoint& globalPos)
{
    if (_menuOpen) return;
    _menuOpen = true;

    // Wait until the right-button event finishes. Opening during release
    // dismisses the menu immediately, so the first click looks like a miss.
    QTimer::singleShot(0, this, [this, globalPos]() {
        QMenu menu(this);
        QAction* exitAction = menu.addAction("Exit data collection");
        connect(exitAction, &QAction::triggered, this, &SimulationPanel::noneOperationRequested);
        QAction* probeAction = menu.addAction("Probe");
        connect(probeAction, &QAction::triggered, this, &SimulationPanel::probeOperationRequested);
        QAction* subtractAction = menu.addAction("Subtraction");
        connect(subtractAction, &QAction::triggered, this,
                &SimulationPanel::subtractionOperationRequested);
        QAction* unionAction = menu.addAction("Union");
        connect(unionAction, &QAction::triggered, this, &SimulationPanel::unionOperationRequested);
        QAction* inspectAction = menu.addAction("Inspection");
        connect(inspectAction, &QAction::triggered, this,
                &SimulationPanel::inspectionOperationRequested);
        menu.addSeparator();
        QAction* clearPathAction = menu.addAction("Clear path line");
        connect(clearPathAction, &QAction::triggered, this, [this]() {
            if (_renderManager) _renderManager->clearTrajectory();
        });
        menu.exec(globalPos);
        _menuOpen = false;
    });
}

void SimulationPanel::drainPoseBuffer()
{
    const std::vector<ToolSample> samples = _log.take();
    if (samples.empty()) return;
    appendSamples(samples);
    applyFifoCap();
    if (_table->rowCount() > 0)
        _table->scrollToBottom();
}

void SimulationPanel::onModeChanged(int index)
{
    const auto mode = static_cast<SimulationMode>(_modeCombo->itemData(index).toInt());
    Parameter::instance().setSimulationMode(mode);
    if (mode != SimulationMode::Interactive) stopPlayback();
    setTableVisibleForMode();
}

void SimulationPanel::endHelixAxisPick()
{
    if (!_pickingHelixAxis) return;
    _pickingHelixAxis = false;
    QApplication::restoreOverrideCursor();
}

void SimulationPanel::beginHelixAxisPick()
{
    if (_pickingHelixAxis) return;
    _pickingHelixAxis = true;
    _reopenHelixDialog = true;
    if (_renderManager) _renderManager->selectWorldAxis(WorldAxisPart::None);
    QApplication::setOverrideCursor(Qt::CrossCursor);
}

void SimulationPanel::reopenHelixDialogIfNeeded()
{
    if (!_reopenHelixDialog) return;
    _reopenHelixDialog = false;
    // Defer so the viewport click finishes before the modal dialog returns.
    QTimer::singleShot(0, this, [this]() { onHelixClicked(); });
}

void SimulationPanel::setHelixAxisFromPick(int axisIndex)
{
    if (axisIndex < 0 || axisIndex > 2) return;
    static const char* axisNames[] = {"X", "Y", "Z"};
    _helixAxis = axisIndex;
    _helixAxisStatus = QString("Selected axis: %1").arg(axisNames[axisIndex]);
    endHelixAxisPick();
    if (_renderManager)
        _renderManager->selectWorldAxis(static_cast<WorldAxisPart>(axisIndex * 2));
    reopenHelixDialogIfNeeded();
}

void SimulationPanel::cancelHelixAxisPick()
{
    if (!_pickingHelixAxis) return;
    static const char* axisNames[] = {"X", "Y", "Z"};
    const char* name =
        (_helixAxis >= 0 && _helixAxis < 3) ? axisNames[_helixAxis] : "X";
    _helixAxisStatus = QString("Axis selection cancelled. Current axis: %1").arg(name);
    endHelixAxisPick();
    reopenHelixDialogIfNeeded();
}

void SimulationPanel::onHelixClicked()
{
    constexpr int SelectAxisResult = 2;

    QDialog dialog(this);
    dialog.setWindowTitle("Helix");
    dialog.setModal(true);

    auto* form = new QFormLayout();
    auto* radiusSpin = new QDoubleSpinBox();
    radiusSpin->setRange(1.0e-6, 1.0e6);
    radiusSpin->setDecimals(4);
    radiusSpin->setSingleStep(0.1);
    radiusSpin->setValue(_helixRadius);
    auto* pitchSpin = new QDoubleSpinBox();
    pitchSpin->setRange(-1.0e6, 1.0e6);
    pitchSpin->setDecimals(4);
    pitchSpin->setSingleStep(0.1);
    pitchSpin->setValue(_helixPitch);
    if (std::fabs(pitchSpin->value()) < 1.0e-6) pitchSpin->setValue(1.0);
    auto* heightSpin = new QDoubleSpinBox();
    heightSpin->setRange(1.0e-6, 1.0e6);
    heightSpin->setDecimals(4);
    heightSpin->setSingleStep(0.1);
    heightSpin->setValue(_helixHeight);
    auto* dtSpin = new QDoubleSpinBox();
    dtSpin->setRange(1.0e-4, 10.0);
    dtSpin->setDecimals(4);
    dtSpin->setSingleStep(0.01);
    dtSpin->setValue(_helixDt);
    dtSpin->setToolTip("Smaller t increment generates more points along the helix.");

    static const char* axisNames[] = {"X", "Y", "Z"};
    const char* axisName =
        (_helixAxis >= 0 && _helixAxis < 3) ? axisNames[_helixAxis] : "X";
    auto* axisButton = new QPushButton(QString("Select axis (%1)…").arg(axisName));
    axisButton->setToolTip("Click to pick X, Y, or Z from the world axes in the viewport.");

    auto* axisStatus = new QLabel(
        _helixAxisStatus.isEmpty() ? QString("Current axis: %1").arg(axisName)
                                   : _helixAxisStatus);
    axisStatus->setWordWrap(true);
    axisStatus->setStyleSheet("color: #c8d2dc;");

    form->addRow("Radius", radiusSpin);
    form->addRow("Pitch", pitchSpin);
    form->addRow("Total height", heightSpin);
    form->addRow("t increment", dtSpin);
    form->addRow("Axis", axisButton);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    QObject::connect(axisButton, &QPushButton::clicked, &dialog, [&]() {
        _helixRadius = radiusSpin->value();
        _helixPitch = pitchSpin->value();
        _helixHeight = heightSpin->value();
        _helixDt = dtSpin->value();
        dialog.done(SelectAxisResult);
    });

    auto* layout = new QVBoxLayout(&dialog);
    layout->addLayout(form);
    layout->addWidget(axisStatus);
    layout->addWidget(buttons);

    const int result = dialog.exec();
    if (result == SelectAxisResult)
    {
        beginHelixAxisPick();
        return;
    }
    if (result != QDialog::Accepted) return;

    _helixRadius = radiusSpin->value();
    _helixPitch = pitchSpin->value();
    _helixHeight = heightSpin->value();
    _helixDt = dtSpin->value();
    if (std::fabs(_helixPitch) < 1.0e-12) return;
    if (_helixDt < 1.0e-12) return;

    // Reset table, then restore pristine cast / cut face so a new helix is not
    // drawn over leftover cuts (same stock wipe as Re-run start).
    onReset();
    if (_renderManager)
    {
        _renderManager->resetBooleanStock();
        _renderManager->resetSweepAnchor();
    }

    const double dt = _helixDt;
    const double r = _helixRadius;
    const double pitch = _helixPitch;
    const double height = _helixHeight;
    const double tEnd = height / std::fabs(pitch);
    const int axis = (_helixAxis >= 0 && _helixAxis < 3) ? _helixAxis : 0;

    vsg::dvec3 axialDir(0.0, 0.0, 0.0);
    axialDir[axis] = 1.0;
    // Orthonormal frame in the plane perpendicular to the helix axis.
    vsg::dvec3 u(0.0, 0.0, 0.0);
    vsg::dvec3 v(0.0, 0.0, 0.0);
    if (axis == 0)
    {
        u = {0.0, 1.0, 0.0};
        v = {0.0, 0.0, 1.0};
    }
    else if (axis == 1)
    {
        u = {0.0, 0.0, 1.0};
        v = {1.0, 0.0, 0.0};
    }
    else
    {
        u = {1.0, 0.0, 0.0};
        v = {0.0, 1.0, 0.0};
    }

    // Start at the low end of the stock along the chosen axis when known.
    double axialStart = 0.0;
    if (_renderManager)
    {
        BoundingBox model;
        if (_renderManager->currentBRep())
            model = BoundingBox::fromBRep(*_renderManager->currentBRep());
        if (model.valid()) axialStart = model.min()[axis];
    }

    vsg::dmat4 modelToWorld{};
    if (_renderManager) modelToWorld = _renderManager->currentFitMatrix();

    auto toWorld = [&](const vsg::dvec3& p) { return modelToWorld * p; };
    auto toWorldDir = [&](const vsg::dvec3& d) {
        const vsg::dvec4 h = modelToWorld * vsg::dvec4(d.x, d.y, d.z, 0.0);
        return vsg::dvec3(h.x, h.y, h.z);
    };

    // Tool spindle along the helix axis.
    const ToolPose axisPose{vsg::dvec3(0.0, 0.0, 0.0), axialDir};
    const ToolSample axisSample = toolSampleFromPose(axisPose);

    std::vector<ToolSample> samples;
    std::vector<vsg::dvec3> path;
    samples.reserve(static_cast<std::size_t>(tEnd / dt) + 2);
    path.reserve(samples.capacity());

    for (double t = 0.0;; t += dt)
    {
        const double useT = (t > tEnd) ? tEnd : t;
        const double axial = pitch * useT; // signed advance along the axis
        const vsg::dvec3 model =
            axialDir * (axialStart + axial) + u * (r * std::cos(useT)) + v * (r * std::sin(useT));
        const vsg::dvec3 modelT =
            axialDir * pitch + u * (-r * std::sin(useT)) + v * (r * std::cos(useT));
        const vsg::dvec3 world = toWorld(model);
        ToolSample sample;
        sample.x = world.x;
        sample.y = world.y;
        sample.z = world.z;
        sample.a = axisSample.a;
        sample.b = axisSample.b;
        sample.c = axisSample.c;
        sample.hasFeed = true;
        sample.feed = toWorldDir(modelT);
        samples.push_back(sample);
        path.push_back(world);
        if (t >= tEnd) break;
    }

    appendSamples(samples);
    applyFifoCap();
    if (_table->rowCount() > 0) _table->scrollToBottom();

    if (_renderManager) _renderManager->setTrajectoryPath(path);
}

void SimulationPanel::setTableVisibleForMode()
{
    const auto mode = Parameter::instance().simulationMode();
    // Pose table + tool library stay available for Interactive and CL data.
    _tableHost->setVisible(mode == SimulationMode::Interactive ||
                           mode == SimulationMode::ClData);
    _clDataHost->setVisible(mode == SimulationMode::ClData);
}

void SimulationPanel::appendSamples(const std::vector<ToolSample>& samples)
{
    for (const ToolSample& sample : samples)
    {
        const int row = _table->rowCount();
        _table->insertRow(row);
        _table->setItem(row, 0, makeCell(sample.x, 8));
        _table->setItem(row, 1, makeCell(sample.y, 8));
        _table->setItem(row, 2, makeCell(sample.z, 8));
        _table->setItem(row, 3, makeCell(sample.a, 2));
        _table->setItem(row, 4, makeCell(sample.b, 2));
        _table->setItem(row, 5, makeCell(sample.c, 2));
        if (sample.hasFeed)
            _pathFeeds.emplace_back(sample.feed);
        else
            _pathFeeds.emplace_back(std::nullopt);
    }
}

void SimulationPanel::appendNullRow()
{
    if (_table->rowCount() == 0) return;
    if (isNullRow(_table->rowCount() - 1)) return;

    const int row = _table->rowCount();
    _table->insertRow(row);
    _table->setItem(row, 0, makeNullCell(true));
    for (int col = 1; col < 6; ++col)
        _table->setItem(row, col, makeNullCell(false));
    _pathFeeds.emplace_back(std::nullopt);
    applyFifoCap();
    _table->scrollToBottom();
}

bool SimulationPanel::isNullRow(int row) const
{
    const QTableWidgetItem* item = _table->item(row, 0);
    return item && item->text() == QLatin1String("Null");
}

void SimulationPanel::applyFifoCap()
{
    // CL data keeps the full path (helix / APT); Interactive still FIFO-caps.
    if (Parameter::instance().simulationMode() == SimulationMode::ClData) return;
    const int extra = _table->rowCount() - maxRows;
    if (extra > 0)
    {
        _table->model()->removeRows(0, extra);
        if (static_cast<int>(_pathFeeds.size()) > extra)
            _pathFeeds.erase(_pathFeeds.begin(), _pathFeeds.begin() + extra);
        else
            _pathFeeds.clear();
    }
}

int SimulationPanel::rowsPerStepFromSlider() const
{
    const int value = _waitSlider ? _waitSlider->value() : 0;
    // Min speed (0) → 1 row; max speed (sliderMax) → 100 rows.
    return 1 + (value * (maxRowsPerStep - 1)) / sliderMax;
}

ToolSample SimulationPanel::sampleAt(int row) const
{
    ToolSample sample;
    auto cell = [this, row](int col) {
        const QTableWidgetItem* item = _table->item(row, col);
        return item ? item->text().toDouble() : 0.0;
    };
    sample.x = cell(0);
    sample.y = cell(1);
    sample.z = cell(2);
    sample.a = cell(3);
    sample.b = cell(4);
    sample.c = cell(5);
    return sample;
}

void SimulationPanel::applyRow(int row)
{
    if (!_renderManager || row < 0 || row >= _table->rowCount()) return;

    _table->selectRow(row);
    if (QTableWidgetItem* item = _table->item(row, 0))
        _table->scrollToItem(item);

    if (isNullRow(row))
        _renderManager->retractToolAndResetSweep();
    else
    {
        const ToolPose pose = toolPoseFromSample(sampleAt(row));
        const vsg::dvec3* along = nullptr;
        vsg::dvec3 feed;
        if (row >= 0 && static_cast<std::size_t>(row) < _pathFeeds.size() &&
            _pathFeeds[static_cast<std::size_t>(row)])
        {
            feed = *_pathFeeds[static_cast<std::size_t>(row)];
            along = &feed;
        }
        _renderManager->setToolPose(pose.position, pose.direction, along);
    }
}

void SimulationPanel::applyRowRange(int fromRow, int toRow)
{
    if (!_renderManager || !_table) return;
    const int n = _table->rowCount();
    if (fromRow < 0 || toRow < fromRow || toRow >= n) return;

    // Null breaks the solid sweep — land on the first null (or stop before it).
    for (int row = fromRow + 1; row <= toRow; ++row)
    {
        if (isNullRow(row))
        {
            toRow = row;
            break;
        }
    }
    if (isNullRow(fromRow) || isNullRow(toRow) || fromRow == toRow)
    {
        applyRow(toRow);
        return;
    }

    std::vector<ToolPose> poses;
    poses.reserve(static_cast<std::size_t>(toRow - fromRow + 1));
    for (int row = fromRow; row <= toRow; ++row)
    {
        ToolPose pose = toolPoseFromSample(sampleAt(row));
        if (row >= 0 && static_cast<std::size_t>(row) < _pathFeeds.size() &&
            _pathFeeds[static_cast<std::size_t>(row)])
            pose.feed = *_pathFeeds[static_cast<std::size_t>(row)];
        poses.push_back(pose);
    }

    _table->selectRow(toRow);
    if (QTableWidgetItem* item = _table->item(toRow, 0))
        _table->scrollToItem(item);

    _renderManager->setToolPosePath(poses);
}

void SimulationPanel::updateRerunButton()
{
    if (_rerunButton)
        _rerunButton->setText(_playing ? QStringLiteral("Pause") : QStringLiteral("Re run"));
}

void SimulationPanel::setCutMeshDisplayForPlayback(bool showMesh)
{
    if (Parameter::instance().cutMeshDisplay() == showMesh) return;
    Parameter::instance().setCutMeshDisplay(showMesh);
    if (_renderManager) _renderManager->refreshCutMeshDisplay();
}

void SimulationPanel::stopPlayback()
{
    _playing = false;
    _paused = false;
    _playRow = 0;
    if (_playTimer) _playTimer->stop();
    updateRerunButton();
    // Do not force cut-mesh display on — that overwrote the checkbox and kept
    // patchCutFace running while the UI still looked unchecked.
}

void SimulationPanel::pausePlayback()
{
    _playing = false;
    _paused = true;
    if (_playTimer) _playTimer->stop();
    updateRerunButton();
}

void SimulationPanel::finishPlayback()
{
    stopPlayback();
    emit noneOperationRequested();
}

void SimulationPanel::onReset()
{
    stopPlayback();
    _table->setRowCount(0);
    _pathFeeds.clear();
    _log.take();
    _lastRecorded.reset();
}

void SimulationPanel::onRerun()
{
    if (_playing)
    {
        pausePlayback();
        return;
    }
    if (!_renderManager || _table->rowCount() < 1) return;

    if (_paused && _playRow >= 0 && _playRow < _table->rowCount())
    {
        _paused = false;
        _playing = true;
        // Keep cut-mesh remesh on (same patchCutFace path as Interactive).
        updateRerunButton();
        if (_playRow + 1 >= _table->rowCount())
        {
            finishPlayback();
            return;
        }
        _playTimer->start(0);
        return;
    }

    if (!prepareRerunTool()) return;

    // Replay must start from the original cast. Recutting already-subtracted
    // stock leaves 1–2 tick islands that draw as orphan Gaussians.
    if (appliesBoolean(Parameter::instance().booleanOp()))
        _renderManager->resetBooleanStock();
    _renderManager->resetSweepAnchor();
    _paused = false;
    _playing = true;
    // Keep cut-mesh remesh on (same patchCutFace path as Interactive).
    _playRow = 0;
    updateRerunButton();
    applyRow(0);

    if (_table->rowCount() == 1)
    {
        finishPlayback();
        return;
    }

    _playTimer->start(0);
}

void SimulationPanel::onPlayStep()
{
    if (!_playing) return;

    const int n = _table->rowCount();
    if (_playRow + 1 >= n)
    {
        finishPlayback();
        return;
    }

    // With cut-mesh remesh on, keep steps Interactive-sized: one row per tick.
    // Large multi-row batches inflate dirty UV windows and hang in patchCutFace.
    // Slider batching stays available for dots-only (mesh off) fast replay.
    const int stride = Parameter::instance().cutMeshDisplay()
                           ? 1
                           : rowsPerStepFromSlider();
    const int toRow = std::min(_playRow + stride, n - 1);
    applyRowRange(_playRow, toRow);
    _playRow = toRow;

    if (_playRow + 1 >= n)
    {
        finishPlayback();
        return;
    }

    _playTimer->start(0);
}

void SimulationPanel::onWaitSliderChanged(int)
{
    // Stride is read each step; no timer restart needed.
}

void SimulationPanel::clearLibrarySelection()
{
    if (!_toolTable) return;
    _toolTable->blockSignals(true);
    _toolTable->clearSelection();
    _toolTable->setCurrentItem(nullptr);
    _toolTable->blockSignals(false);
}

void SimulationPanel::updateLibraryEntry(int toolType, double radius, double cuttingLength,
                                         double shankLength, double shankRadius, double tipWidth,
                                         double shoulderWidth, double taperHeight,
                                         double shoulderHeight)
{
    if (!_toolTable || toolType == static_cast<int>(ToolType::None)) return;
    if (!(radius > 0.0) || !(cuttingLength > 0.0) || !(shankLength > 0.0) ||
        !(shankRadius > 0.0))
        return;

    int matchRow = -1;
    const bool blocked = _toolTable->blockSignals(true);
    for (int row = 0; row < _toolTable->rowCount(); ++row)
    {
        QTableWidgetItem* idItem = _toolTable->item(row, 0);
        if (!idItem || idItem->data(Qt::UserRole).toInt() != toolType) continue;
        matchRow = row;
        if (toolType == static_cast<int>(ToolType::GrindingWheel))
        {
            if (tipWidth > 0.0) idItem->setData(Qt::UserRole + 1, tipWidth);
            if (shoulderWidth > 0.0) idItem->setData(Qt::UserRole + 2, shoulderWidth);
            if (taperHeight > 0.0) idItem->setData(Qt::UserRole + 3, taperHeight);
            if (shoulderHeight >= 0.0) idItem->setData(Qt::UserRole + 4, shoulderHeight);
            const double totalH =
                idItem->data(Qt::UserRole + 3).toDouble() + idItem->data(Qt::UserRole + 4).toDouble();
            if (totalH > 0.0)
            {
                radius = totalH;
                cuttingLength = totalH;
            }
        }
        auto setNumber = [this, row](int col, double value) {
            QTableWidgetItem* cell = _toolTable->item(row, col);
            if (!cell)
            {
                cell = makeEditableCell(value, 6);
                _toolTable->setItem(row, col, cell);
            }
            else
            {
                cell->setText(QString::number(value, 'f', 6));
            }
        };
        setNumber(2, radius);
        setNumber(3, cuttingLength);
        setNumber(4, shankLength);
        setNumber(5, shankRadius);
        break;
    }
    _toolTable->blockSignals(blocked);

    if (matchRow >= 0 && matchRow == _toolTable->currentRow() &&
        !_toolTable->selectedItems().isEmpty())
        applySelectedLibraryTool();
}

void SimulationPanel::onToolLibrarySelectionChanged()
{
    applySelectedLibraryTool();
}

bool SimulationPanel::applySelectedLibraryTool()
{
    if (!_toolTable) return false;

    const int row = _toolTable->currentRow();
    if (row < 0 || _toolTable->selectedItems().isEmpty()) return false;

    const QTableWidgetItem* idItem = _toolTable->item(row, 0);
    const QTableWidgetItem* radiusItem = _toolTable->item(row, 2);
    const QTableWidgetItem* cuttingItem = _toolTable->item(row, 3);
    const QTableWidgetItem* shankItem = _toolTable->item(row, 4);
    const QTableWidgetItem* shankRadiusItem = _toolTable->item(row, 5);
    if (!idItem || !radiusItem || !cuttingItem) return false;

    const auto type = static_cast<ToolType>(idItem->data(Qt::UserRole).toInt());
    if (type == ToolType::None) return false;

    const double radius = radiusItem->text().toDouble();
    const double cuttingLength = cuttingItem->text().toDouble();
    const double shankLength = shankItem ? shankItem->text().toDouble() : cuttingLength;
    const double shankRadius =
        shankRadiusItem ? shankRadiusItem->text().toDouble() : radius;

    Parameter::instance().setToolType(type);
    Parameter::instance().setToolRadius(radius);
    Parameter::instance().setToolLength(cuttingLength);
    Parameter::instance().setToolShankRadius(shankRadius);
    Parameter::instance().setToolShankLength(shankLength);
    if (type == ToolType::GrindingWheel)
    {
        const double tipWidth = idItem->data(Qt::UserRole + 1).toDouble();
        const double shoulderWidth = idItem->data(Qt::UserRole + 2).toDouble();
        const double taperHeight = idItem->data(Qt::UserRole + 3).toDouble();
        const double shoulderHeight = idItem->data(Qt::UserRole + 4).toDouble();
        if (tipWidth > 0.0) Parameter::instance().setToolWheelTipWidth(tipWidth);
        if (shoulderWidth > 0.0) Parameter::instance().setToolWheelShoulderWidth(shoulderWidth);
        if (taperHeight > 0.0) Parameter::instance().setToolWheelTaperHeight(taperHeight);
        if (shoulderHeight >= 0.0) Parameter::instance().setToolWheelShoulderHeight(shoulderHeight);
        const double totalH = Parameter::instance().toolWheelTaperHeight() +
                              Parameter::instance().toolWheelShoulderHeight();
        if (totalH > 0.0)
        {
            Parameter::instance().setToolRadius(totalH);
            Parameter::instance().setToolLength(totalH);
        }
    }
    if (_renderManager)
    {
        _renderManager->setToolType(type);
        _renderManager->updateToolGeometry();
    }
    emit toolLibraryApplied(static_cast<int>(type), radius, cuttingLength, shankLength,
                            shankRadius);
    return true;
}

bool SimulationPanel::prepareRerunTool()
{
    if (applySelectedLibraryTool()) return true;
    return Parameter::instance().toolType() != ToolType::None;
}

void SimulationPanel::onToolLibraryDoubleClicked(int row, int)
{
    if (!_toolTable || row < 0 || row >= _toolTable->rowCount()) return;
    const QTableWidgetItem* idItem = _toolTable->item(row, 0);
    const QTableWidgetItem* radiusItem = _toolTable->item(row, 2);
    const QTableWidgetItem* cuttingItem = _toolTable->item(row, 3);
    const QTableWidgetItem* shankItem = _toolTable->item(row, 4);
    const QTableWidgetItem* shankRadiusItem = _toolTable->item(row, 5);
    if (!idItem || !radiusItem || !cuttingItem || !shankItem || !shankRadiusItem) return;

    emit toolLibraryPreviewRequested(idItem->data(Qt::UserRole).toInt(),
                                     radiusItem->text().toDouble(),
                                     cuttingItem->text().toDouble(),
                                     shankItem->text().toDouble(),
                                     shankRadiusItem->text().toDouble(),
                                     idItem->data(Qt::UserRole + 1).toDouble(),
                                     idItem->data(Qt::UserRole + 2).toDouble(),
                                     idItem->data(Qt::UserRole + 3).toDouble(),
                                     idItem->data(Qt::UserRole + 4).toDouble());
}

void SimulationPanel::fillToolLibrary()
{
    if (!_toolTable) return;

    const Parameter& store = Parameter::instance();
    const double radius = store.toolRadius();
    const double length = store.toolLength();

    struct Seed
    {
        int id;
        ToolType type;
    };
    const Seed seeds[] = {
        {1, ToolType::BullNose},
        {2, ToolType::FlatNose},
        {3, ToolType::BallNose},
        {4, ToolType::Sphere},
        {5, ToolType::GrindingWheel},
    };

    const double tipWidth = store.toolWheelTipWidth() > 0.0 ? store.toolWheelTipWidth() : 0.01;
    const double shoulderWidth =
        store.toolWheelShoulderWidth() > 0.0 ? store.toolWheelShoulderWidth() : 0.06;
    const double taperHeight =
        store.toolWheelTaperHeight() > 0.0 ? store.toolWheelTaperHeight() : 0.035;
    const double shoulderHeight =
        store.toolWheelShoulderHeight() >= 0.0 ? store.toolWheelShoulderHeight() : 0.015;

    _toolTable->blockSignals(true);
    _toolTable->setRowCount(0);
    for (const Seed& seed : seeds)
    {
        const int row = _toolTable->rowCount();
        _toolTable->insertRow(row);
        auto* idItem = makeTextCell(QString::number(seed.id));
        idItem->setData(Qt::UserRole, static_cast<int>(seed.type));
        if (seed.type == ToolType::GrindingWheel)
        {
            idItem->setData(Qt::UserRole + 1, tipWidth);
            idItem->setData(Qt::UserRole + 2, shoulderWidth);
            idItem->setData(Qt::UserRole + 3, taperHeight);
            idItem->setData(Qt::UserRole + 4, shoulderHeight);
        }
        _toolTable->setItem(row, 0, idItem);
        _toolTable->setItem(row, 1, makeTextCell(QString::fromLatin1(toolTypeLabel(seed.type)),
                                                 Qt::AlignLeft));
        const double shankRadius =
            (seed.type == ToolType::Sphere) ? radius * 0.6 : radius * 1.2;
        const double rowLength =
            (seed.type == ToolType::GrindingWheel) ? (taperHeight + shoulderHeight) : length;
        const double rowRadius =
            (seed.type == ToolType::GrindingWheel) ? rowLength : radius;
        _toolTable->setItem(row, 2, makeEditableCell(rowRadius, 6));
        _toolTable->setItem(row, 3, makeEditableCell(rowLength, 6));
        _toolTable->setItem(row, 4, makeEditableCell(length, 6));
        _toolTable->setItem(row, 5, makeEditableCell(shankRadius, 6));
    }
    _toolTable->blockSignals(false);
}

void SimulationPanel::onToolLibraryCellChanged(QTableWidgetItem* item)
{
    if (!_toolTable || !item) return;
    const int row = item->row();
    const int col = item->column();
    if (row < 0 || col < 2 || col > 5) return;

    double value = 0.0;
    if (!parsePositive(item->text(), &value))
    {
        const Parameter& store = Parameter::instance();
        if (col == 2) value = store.toolRadius() > 0.0 ? store.toolRadius() : 0.05;
        else if (col == 3 || col == 4)
        {
            const double cutting = store.toolLength();
            const double radius = store.toolRadius();
            value = cutting > 0.0 ? cutting : (radius > 0.0 ? radius * 2.8 : 0.14);
        }
        else
        {
            const double radius = store.toolRadius();
            value = radius > 0.0 ? radius : 0.05;
        }
    }

    const bool blocked = _toolTable->blockSignals(true);
    item->setText(QString::number(value, 'f', 6));
    _toolTable->blockSignals(blocked);

    if (row == _toolTable->currentRow() && !_toolTable->selectedItems().isEmpty())
        applySelectedLibraryTool();
}

} // namespace app
