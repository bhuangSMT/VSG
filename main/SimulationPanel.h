// SimulationPanel - right-hand Interactive / CL / NC panel and pose table.
#pragma once

#include <memory>
#include <optional>
#include <vector>

#include <QtCore/QPoint>
#include <QtCore/QString>
#include <QtGui/QColor>
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
class QTableWidgetItem;
class QTimer;

namespace app
{

class RenderManager;

class SimulationPanel : public QWidget
{
    Q_OBJECT

public:
    static constexpr int maxRows = 5000;
    static constexpr int sliderMax = 100;
    static constexpr int maxRowsPerStep = 100;

    explicit SimulationPanel(QWidget* parent = nullptr);

    void setRenderManager(std::shared_ptr<RenderManager> renderManager);

    bool isPlaying() const { return _playing; }

    // Push a sample from the mouse-move path. Queues one drain event if none
    // is already pending. No-op unless Interactive, Ray / Ray-GS, and Operation
    // records a tool path (not None or Inspection).
    void record(const ToolPose& pose);

    void popupExitCollectionMenu(const QPoint& globalPos);
    void notifyBooleanOp(BooleanOp op);
    void clearLibrarySelection();
    // Upsert a Tool manager edit into the Simulation library table by tool id.
    // Re-applies the cutter when that row is the current selection.
    void updateLibraryEntry(int toolId, int toolType, double radius, double cuttingLength,
                            double shankLength, double shankRadius, double tipWidth = 0.0,
                            double shoulderWidth = 0.0, double taperHeight = 0.0,
                            double shoulderHeight = 0.0);
    void updateLibraryColor(int toolId, int toolType, const QColor& color);
    void removeLibraryTool(int toolId);

    // Helix axis pick: Axis button in the Helix dialog arms this; ToolTracker
    // completes/cancels it and the dialog reopens.
    bool isPickingHelixAxis() const { return _pickingHelixAxis; }
    void setHelixAxisFromPick(int axisIndex);
    void cancelHelixAxisPick();

signals:
    void noneOperationRequested();
    void probeOperationRequested();
    void subtractionOperationRequested();
    void unionOperationRequested();
    void inspectionOperationRequested();
    void toolLibraryPreviewRequested(int toolId, int toolType, double radius, double cuttingLength,
                                     double shankLength, double shankRadius, double tipWidth,
                                     double shoulderWidth, double taperHeight,
                                     double shoulderHeight);
    void toolLibraryApplied(int toolType, double radius, double cuttingLength, double shankLength,
                            double shankRadius);

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
    void onToolLibraryDoubleClicked(int row, int column);
    void onToolLibrarySelectionChanged();
    void onToolLibraryCellChanged(QTableWidgetItem* item);
    void onHelixClicked();

private:
    void applyFifoCap();
    void appendSamples(const std::vector<ToolSample>& samples);
    void appendNullRow();
    void setTableVisibleForMode();
    void stopPlayback();
    void pausePlayback();
    void finishPlayback();
    // Ensure mesh display matches showMesh (used on pause/stop to keep remesh on).
    void setCutMeshDisplayForPlayback(bool showMesh);
    void updateRerunButton();
    void beginHelixAxisPick();
    void endHelixAxisPick();
    void reopenHelixDialogIfNeeded();
    void applyRow(int row);
    // Advance from fromRow to toRow (inclusive), building one multi-station
    // swept volume through the table poses when toRow > fromRow.
    void applyRowRange(int fromRow, int toRow);
    ToolSample sampleAt(int row) const;
    bool isNullRow(int row) const;
    int rowsPerStepFromSlider() const;
    void fillToolLibrary();
    int libraryRowForId(int toolId) const;
    bool applySelectedLibraryTool();
    bool prepareRerunTool();

    QComboBox* _modeCombo = nullptr;
    QWidget* _clDataHost = nullptr;
    QPushButton* _helixButton = nullptr;
    QPushButton* _importAptButton = nullptr;
    QTableWidget* _table = nullptr;
    QWidget* _tableHost = nullptr;
    QPushButton* _rerunButton = nullptr;
    QPushButton* _resetButton = nullptr;
    QSlider* _waitSlider = nullptr;
    QTableWidget* _toolTable = nullptr;
    QTimer* _playTimer = nullptr;
    ToolPoseLog _log;
    std::optional<ToolSample> _lastRecorded;
    std::shared_ptr<RenderManager> _renderManager;
    bool _menuOpen = false;
    bool _playing = false;
    bool _paused = false;
    int _playRow = 0;
    double _helixRadius = 1.0;
    double _helixPitch = 1.0;
    double _helixHeight = 10.0;
    double _helixDt = 0.1;
    int _helixAxis = 0; // 0=X, 1=Y, 2=Z
    // Analytic helix tangents, one per table row (nullopt for Interactive / APT).
    std::vector<std::optional<vsg::dvec3>> _pathFeeds;
    QString _helixAxisStatus;
    bool _pickingHelixAxis = false;
    bool _reopenHelixDialog = false;
};

} // namespace app
