#include "SimulationPanel.h"

#include <cmath>

#include <QtCore/QEvent>
#include <QtCore/QTimer>
#include <QtGui/QContextMenuEvent>
#include <QtGui/QFont>
#include <QtGui/QHideEvent>
#include <QtGui/QStandardItemModel>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMenu>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTableWidgetItem>
#include <QtWidgets/QVBoxLayout>

#include "BooleanOp.h"
#include "Parameter.h"
#include "RenderManager.h"
#include "SimulationMode.h"
#include "ViewMode.h"

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

QTableWidgetItem* makeNullCell(bool label)
{
    auto* item = new QTableWidgetItem(label ? QStringLiteral("Null") : QString());
    item->setFlags(item->flags() & ~Qt::ItemIsEditable);
    item->setTextAlignment(Qt::AlignCenter);
    return item;
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
    setFixedWidth(300);

    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(8, 8, 8, 8);
    layout->setSpacing(6);

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
        if (QStandardItem* cl = model->item(1)) cl->setEnabled(false);
        if (QStandardItem* nc = model->item(2)) nc->setEnabled(false);
    }
    layout->addWidget(_modeCombo);

    _tableHost = new QWidget();
    auto* tableLayout = new QVBoxLayout(_tableHost);
    tableLayout->setContentsMargins(0, 0, 0, 0);
    tableLayout->setSpacing(6);

    _table = new QTableWidget(0, 6);
    _table->setHorizontalHeaderLabels({"X", "Y", "Z", "A", "B", "C"});
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setSelectionMode(QAbstractItemView::SingleSelection);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _table->verticalHeader()->setVisible(false);
    _table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    tableLayout->addWidget(_table, 1);

    auto* buttonRow = new QHBoxLayout();
    buttonRow->setContentsMargins(0, 0, 0, 0);
    buttonRow->setSpacing(6);
    _rerunButton = new QPushButton("Rerun");
    _resetButton = new QPushButton("Reset");
    buttonRow->addWidget(_rerunButton);
    buttonRow->addWidget(_resetButton);
    tableLayout->addLayout(buttonRow);

    _waitSlider = new QSlider(Qt::Horizontal);
    _waitSlider->setRange(0, sliderMax);
    _waitSlider->setValue(sliderMax / 2);
    _waitSlider->setToolTip("Step wait: 0 = 2 s, 100 = 0 s");
    tableLayout->addWidget(_waitSlider);

    layout->addWidget(_tableHost, 1);
    layout->addStretch(1);

    _playTimer = new QTimer(this);
    _playTimer->setSingleShot(true);

    connect(_modeCombo, &QComboBox::currentIndexChanged, this, &SimulationPanel::onModeChanged);
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

void SimulationPanel::setTableVisibleForMode()
{
    const bool interactive =
        Parameter::instance().simulationMode() == SimulationMode::Interactive;
    _tableHost->setVisible(interactive);
}

void SimulationPanel::appendSamples(const std::vector<ToolSample>& samples)
{
    for (const ToolSample& sample : samples)
    {
        const int row = _table->rowCount();
        _table->insertRow(row);
        _table->setItem(row, 0, makeCell(sample.x, 4));
        _table->setItem(row, 1, makeCell(sample.y, 4));
        _table->setItem(row, 2, makeCell(sample.z, 4));
        _table->setItem(row, 3, makeCell(sample.a, 2));
        _table->setItem(row, 4, makeCell(sample.b, 2));
        _table->setItem(row, 5, makeCell(sample.c, 2));
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
    const int extra = _table->rowCount() - maxRows;
    if (extra > 0) _table->model()->removeRows(0, extra);
}

int SimulationPanel::waitMsFromSlider() const
{
    const int value = _waitSlider ? _waitSlider->value() : 0;
    return (maxWaitMs * (sliderMax - value)) / sliderMax;
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
        _renderManager->setToolPose(pose.position, pose.direction);
    }
}

void SimulationPanel::updateRerunButton()
{
    if (_rerunButton)
        _rerunButton->setText(_playing ? QStringLiteral("Pause") : QStringLiteral("Rerun"));
}

void SimulationPanel::stopPlayback()
{
    _playing = false;
    _paused = false;
    _playRow = 0;
    if (_playTimer) _playTimer->stop();
    updateRerunButton();
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
    if (Parameter::instance().toolType() == ToolType::None) return;

    if (_paused && _playRow >= 0 && _playRow < _table->rowCount())
    {
        _paused = false;
        _playing = true;
        updateRerunButton();
        if (_playRow + 1 >= _table->rowCount())
        {
            finishPlayback();
            return;
        }
        _playTimer->start(waitMsFromSlider());
        return;
    }

    // Replay must start from the original cast. Recutting already-subtracted
    // stock leaves 1–2 tick islands that draw as orphan Gaussians.
    if (appliesBoolean(Parameter::instance().booleanOp()))
        _renderManager->resetBooleanStock();
    _renderManager->resetSweepAnchor();
    _paused = false;
    _playing = true;
    _playRow = 0;
    updateRerunButton();
    applyRow(0);

    if (_table->rowCount() == 1)
    {
        finishPlayback();
        return;
    }

    _playTimer->start(waitMsFromSlider());
}

void SimulationPanel::onPlayStep()
{
    if (!_playing) return;

    ++_playRow;
    if (_playRow >= _table->rowCount())
    {
        finishPlayback();
        return;
    }

    applyRow(_playRow);
    if (_playRow + 1 >= _table->rowCount())
    {
        finishPlayback();
        return;
    }

    _playTimer->start(waitMsFromSlider());
}

void SimulationPanel::onWaitSliderChanged(int)
{
    if (!_playing || !_playTimer->isActive()) return;
    _playTimer->start(waitMsFromSlider());
}

} // namespace app
