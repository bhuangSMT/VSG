// SimulationPanel - right-hand Interactive / CL / NC panel and pose table.
#pragma once

#include <memory>
#include <optional>

#include <QtCore/QElapsedTimer>
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
    // is not None.
    void record(const ToolPose& pose);

    void popupExitCollectionMenu(const QPoint& globalPos);
    void notifyBooleanOp(BooleanOp op);

signals:
    void noneOperationRequested();
    void probeOperationRequested();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;
    void hideEvent(QHideEvent* event) override;

private slots:
    void drainPoseBuffer();
    void onModeChanged(int index);
    void onRerun();
    void onPlayStep();
    void onWaitSliderChanged(int value);

private:
    void applyFifoCap();
    void appendSamples(const std::vector<ToolSample>& samples);
    void appendNullRow();
    void setTableVisibleForMode();
    void stopPlayback();
    void finishPlayback();
    void applyRow(int row);
    ToolSample sampleAt(int row) const;
    bool isNullRow(int row) const;
    int waitMsFromSlider() const;

    QComboBox* _modeCombo = nullptr;
    QTableWidget* _table = nullptr;
    QWidget* _tableHost = nullptr;
    QPushButton* _rerunButton = nullptr;
    QSlider* _waitSlider = nullptr;
    QTimer* _playTimer = nullptr;
    ToolPoseLog _log;
    std::optional<ToolSample> _lastRecorded;
    std::shared_ptr<RenderManager> _renderManager;
    QElapsedTimer _menuGuard;
    bool _playing = false;
    int _playRow = 0;
};

} // namespace app
