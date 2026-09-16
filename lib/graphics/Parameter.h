// Parameter - the single store of application settings.
//
// Every control in the UI commits its value here first; the code that acts on a
// setting then reads it back from this instance rather than from the widget.
// That keeps the widgets as pure input and gives the rest of the application
// one place to look for the current state.
//
// The instance is created when the application starts and lives until it exits.
#pragma once

#include <string>

#include "Ray.h"
#include "BooleanOp.h"
#include "SimulationMode.h"
#include "ToolType.h"
#include "ViewMode.h"

namespace app
{

class Parameter
{
public:
    // The one instance. Constructed on first call, which main() makes during
    // application start-up so that the store exists before any UI does.
    static Parameter& instance();

    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;

    // How the model is drawn.
    ViewMode viewMode() const { return _viewMode; }
    void setViewMode(ViewMode mode) { _viewMode = mode; }

    // Grid spacing used when casting rays, per axis.
    const Point3d& rayResolution() const { return _rayResolution; }
    void setRayResolution(const Point3d& resolution) { _rayResolution = resolution; }

    // Steps across the bounding box used to seed rayResolution for a new model.
    int rayDivisions() const { return _rayDivisions; }
    void setRayDivisions(int divisions) { _rayDivisions = divisions; }

    // Whether the viewer redraws continuously or only on request.
    bool continuousUpdate() const { return _continuousUpdate; }
    void setContinuousUpdate(bool enabled) { _continuousUpdate = enabled; }

    // Which cutter follows the mouse over the model.
    ToolType toolType() const { return _toolType; }
    void setToolType(ToolType type) { _toolType = type; }

    // Cutter radius in model space. Seeded to 5% of the BRep bounding-box
    // diagonal when a model is loaded; the UI commits edits here before the
    // tool mesh is rebuilt.
    double toolRadius() const { return _toolRadius; }
    void setToolRadius(double radius) { _toolRadius = radius; }

    // Cutting length in model space (above the tip geometry). Default is
    // 2.8 × toolRadius when a model is loaded.
    double toolLength() const { return _toolLength; }
    void setToolLength(double length) { _toolLength = length; }

    // Display-only shank in model space. Zero hides the shank. Swept volume
    // and boolean always use the cutter mesh, never this cylinder.
    double toolShankRadius() const { return _toolShankRadius; }
    void setToolShankRadius(double radius) { _toolShankRadius = radius; }
    double toolShankLength() const { return _toolShankLength; }
    void setToolShankLength(double length) { _toolShankLength = length; }

    // Grinding wheel: isosceles-triangle vertex angle at the outer rim (degrees).
    double toolVertexAngleDeg() const { return _toolVertexAngleDeg; }
    void setToolVertexAngleDeg(double degrees) { _toolVertexAngleDeg = degrees; }

    // Whether the swept-volume mesh is drawn in the scene. Generation always
    // runs while a tool is active; this only gates the VSG node.
    bool sweptVolume() const { return _sweptVolume; }
    void setSweptVolume(bool enabled) { _sweptVolume = enabled; }

    // When true, each new sweep segment replaces the previous one so only the
    // latest segment is kept and drawn.
    bool showLastSweptVolumeOnly() const { return _showLastSweptVolumeOnly; }
    void setShowLastSweptVolumeOnly(bool enabled) { _showLastSweptVolumeOnly = enabled; }

    // Ray-GS cut face: quad mesh overlay when true; cut-tagged splat dots only
    // when false (default).
    bool cutMeshDisplay() const { return _cutMeshDisplay; }
    void setCutMeshDisplay(bool enabled) { _cutMeshDisplay = enabled; }

    // How the swept volume is combined with the current RayModel.
    // Inspection is a non-destructive preview against the original cast.
    BooleanOp booleanOp() const { return _booleanOp; }
    void setBooleanOp(BooleanOp op) { _booleanOp = op; }

    // How the right-hand Simulation panel is driven. Interactive is the only
    // implemented mode; ClData / NcMachining are reserved.
    SimulationMode simulationMode() const { return _simulationMode; }
    void setSimulationMode(SimulationMode mode) { _simulationMode = mode; }

    // The last file the user imported; also where the file dialog reopens.
    const std::string& lastImportPath() const { return _lastImportPath; }
    void setLastImportPath(std::string path) { _lastImportPath = std::move(path); }

private:
    Parameter() = default;

    ViewMode _viewMode = ViewMode::Facet;
    Point3d _rayResolution{0.000625, 0.000625, 0.000625};
    int _rayDivisions = 1600;
    bool _continuousUpdate = true;
    ToolType _toolType = ToolType::None;
    double _toolRadius = 0.05;
    double _toolLength = 0.05 * 2.8;
    double _toolShankRadius = 0.0;
    double _toolShankLength = 0.0;
    double _toolVertexAngleDeg = 60.0;
    bool _sweptVolume = false;
    bool _showLastSweptVolumeOnly = true;
    bool _cutMeshDisplay = false;
    BooleanOp _booleanOp = BooleanOp::None;
    SimulationMode _simulationMode = SimulationMode::Interactive;
    std::string _lastImportPath;
};

} // namespace app
