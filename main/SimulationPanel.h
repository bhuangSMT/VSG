// SimulationPanel - right-hand Interactive / CL / NC panel and pose table.
#pragma once

#include <memory>
#include <optional>

#include <QtCore/QPoint>
#include <QtWidgets/QWidget>

#include "BooleanOp.h"
#include "SweptVolume.h"
#include "ToolPoseLog.h"

class QComboBox;
class QEvent;
class QHideEvent;
class QPushButton;
class QSlider;
class QTableWidget;
class QTimer;

namespace app
{

class RenderManager;

class SimulationPanel : public QWidget
{
    Q_OBJECT

public:
    static constexpr int maxRows = 500;
    static constexpr int sliderMax = 100;
    static constexpr int maxWaitMs = 2000;

    explicit SimulationPanel(QWidget* parent = nullptr);

    void setRenderManager(std::shared_ptr<RenderManager> renderManager);

    bool isPlaying() const { return _playing; }

    // Push a sample from the mouse-move path. Queues one drain event if none
    // is already pending. No-op unless Interactive, Ray / Ray-GS, and Operation
    // records a tool path (not None or Inspection).
    void record(const ToolPose& pose);

    void popupExitCollectionMenu(const QPoint& globalPos);
    void notifyBooleanOp(BooleanOp op);

signals:
    void noneOperationRequested();
    void probeOperationRequested();
    void subtractionOperationRequested();
    void unionOperationRequested();
    void inspectionOperationRequested();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private slots:
    void drainPoseBuffer();
    void onModeChanged(int index);
    void onRerun();
    void onReset();
    void onPlayStep();
    void onWaitSliderChanged(int value);

private:
    void applyFifoCap();
    void appendSamples(const std::vector<ToolSample>& samples);
    void appendNullRow();
    void setTableVisibleForMode();
    void stopPlayback();
    void pausePlayback();
    void finishPlayback();
    void updateRerunButton();
    void applyRow(int row);
    ToolSample sampleAt(int row) const;
    bool isNullRow(int row) const;
    int waitMsFromSlider() const;

    QComboBox* _modeCombo = nullptr;
    QTableWidget* _table = nullptr;
    QWidget* _tableHost = nullptr;
    QPushButton* _rerunButton = nullptr;
    QPushButton* _resetButton = nullptr;
    QSlider* _waitSlider = nullptr;
    QTimer* _playTimer = nullptr;
    ToolPoseLog _log;
    std::optional<ToolSample> _lastRecorded;
    std::shared_ptr<RenderManager> _renderManager;
    bool _menuOpen = false;
    bool _playing = false;
    bool _paused = false;
    int _playRow = 0;
};

} // namespace app
