// RenderManager - the handshake between application geometry and the VSG scene.
//
// Owns the scene root that the command graph renders, converts a BRep (or the
// RayModel cast through it) into a renderable VSG subgraph, compiles that
// subgraph for the already-running viewer and swaps it into the scene.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <vsg/all.h>

#include <vsgQt/Viewer.h>

#include "BRep.h"
#include "BoundingBox.h"
#include "BooleanOp.h"
#include "GaussianSplat.h"
#include "GaussianSplatCache.h"
#include "RayModel.h"
#include "SweptVolume.h"
#include "ToolType.h"
#include "ViewMode.h"
#include "WorldAxes.h"

namespace app
{

class RenderManager
{
public:
    // scene is the root Group referenced by the viewer's command graph; the
    // RenderManager adds/removes its children.
    RenderManager(vsg::ref_ptr<vsgQt::Viewer> viewer,
                  vsg::ref_ptr<vsg::Group> scene,
                  vsg::ref_ptr<vsg::Options> options);
    ~RenderManager();

    // RGB XYZ gizmo at the world origin (tube + cone), kept in its own scene
    // subgraph with one child group per axis. Sized from the stock AABB.
    // Clickable via pickWorldAxis / selectWorldAxis.
    void showWorldAxes(bool show = true);
    bool worldAxesVisible() const { return _axesNode != nullptr; }
    std::optional<WorldAxisPart> pickWorldAxis(const vsg::Camera& camera, int32_t x,
                                               int32_t y) const;
    void selectWorldAxis(WorldAxisPart part);
    WorldAxisPart selectedWorldAxis() const { return _axesSelected; }

    // Replace everything currently in the scene with the given BRep. Any ray
    // model from a previous BRep is discarded.
    // Throws std::runtime_error if the BRep has no drawable triangles or the
    // graphics pipeline cannot be created.
    void showBRep(const BRep& brep);

    // Add the given BRep alongside whatever is already in the scene.
    void addBRep(const BRep& brep);

    // Remove all geometry from the scene.
    void clear();

    // The BRep currently on display, if any. Callers build a RayModel from it
    // and hand the result back via setRayModel().
    const std::optional<BRep>& currentBRep() const { return _current; }

    // Take ownership of the rays cast through the current BRep and keep them
    // under the resolution they were cast at, so that coming back to this
    // resolution later does not need a recast. Redraws immediately when a ray
    // view mode is active.
    void setRayModel(RayModel model);

    // Draw the rays cast at this resolution if they are still held. Returns
    // false when nothing is held for it, in which case the caller has to cast a
    // model and hand it over via setRayModel().
    bool useCachedRayModel(const Point3d& resolution);

    // The rays currently being drawn, or null if none have been cast.
    const RayModel* rayModel() const { return _rayModel; }

    // Build a renderable subgraph without attaching it.
    vsg::ref_ptr<vsg::Node> createNode(const BRep& brep) const;
    vsg::ref_ptr<vsg::Node> createRayNode(const RayModel& rayModel) const;
    vsg::ref_ptr<vsg::Node> createSplatNode(const RayModel& rayModel) const;

    // Switch how geometry is drawn. The BRep last passed to showBRep() is
    // rebuilt in the new mode, so the change is visible immediately.
    void setViewMode(ViewMode mode);
    ViewMode viewMode() const { return _viewMode; }

    // Which cutter follows the mouse. None removes the tool from the scene.
    void setToolType(ToolType type);
    ToolType toolType() const { return _toolType; }

    // Rebuild the cutter (and display shank, if Parameter has shank size)
    // from Parameter. Swept volume still uses the cutter only. Keeps the
    // existing pose when the tool was already on screen.
    void updateToolGeometry();

    // Place the active tool at position with axis along direction (world space).
    // Position is the sphere centre for ball nose / sphere, the fillet-torus
    // centre for bull nose, and the tip for flat nose. The sweep still uses
    // the tip (shifted down the axis by toolCenterOffset).
    void setToolPose(const vsg::dvec3& position, const vsg::dvec3& direction);

    // Advance along a polyline of reference poses (same convention as
    // setToolPose). Builds one swept solid through all stations (caps at the
    // ends only for grinding), places the tool at the last pose, and booleans
    // once. Requires at least two poses.
    void setToolPosePath(const std::vector<ToolPose>& referencePoses);

    // Place the cutter tip at pose.position with axis pose.direction. Unlike
    // setToolPose, position is always the tip (no ball-nose centre offset).
    // Sweeps from the previous tip and booleans with the current operation.
    void setToolTipPose(const ToolPose& pose);

    // Drop the recorded sweep path so the next setToolTipPose only seeds the
    // start pose. Does not revert boolean stock.
    void resetSweepAnchor();

    // Throw away accumulated cuts and show the original cast. Rerun uses this
    // so replay does not recut leftover slivers into orphan Gaussians.
    void resetBooleanStock();

    // Move the cutter along its +axis until it is outside the stock AABB,
    // without boolean. Clears the sweep last-pose so the next setToolTipPose
    // only seeds the start.
    void retractToolAndResetSweep();

    // Tip and axis last written by setToolPose, if a cutter is active.
    const std::optional<ToolPose>& lastToolPose() const { return _lastToolPose; }

    // Mouse / table location: centre for ball, sphere, and bull; tip otherwise.
    std::optional<ToolPose> lastReferencePose() const;

    // Show or hide the swept-volume drawable. Does not start/stop recording.
    void setSweptVolumeVisible(bool visible);
    bool sweptVolumeVisible() const { return _showSweptVolume; }

    // Rebuild Ray-GS after Parameter::cutMeshDisplay() changes. A live GPU cut
    // mesh is kept across playback toggles until reset or a dirty patch.
    void refreshCutMeshDisplay();

    // Drop the accumulated swept-volume mesh (CPU + scene node).
    void clearSweptVolume();

    // Drop the tool-tip polyline node. The next non-None pose starts a new path.
    void clearTrajectory();

    // Replace the path line with an explicit polyline (model/world positions as
    // drawn). Ignores boolean-op / tool-type gates used by appendToolTrajectory.
    void setTrajectoryPath(const std::vector<vsg::dvec3>& points);

    // Fit matrix for the currently displayed stock (BRep / ray model), or identity.
    vsg::dmat4 currentFitMatrix() const;

    // Apply Parameter::booleanOp() using the current SweptVolume. Subtraction
    // and Union are cumulative: the input is the previous boolean result (or
    // the original cast on the first cut). Inspection clones session stock
    // (_booleanRayModel if present, else the original cast) every move and
    // subtracts only the current cutter.
    // None stops further cuts but keeps the current RayModel as displayed.
    void setBooleanOp(BooleanOp op);
    void applyBooleanToRayModel();

    // Hollow the current simulation stock by thickness (model units). On the
    // first Shell, caches a deep clone for Cancel / re-shell. On later Shell
    // OK, restores that cache first so the new thickness replaces the old
    // walls instead of stacking. Returns false if there is no stock or
    // thickness is invalid.
    bool shellStock(double thickness);
    // Restore stock from the pre-shell cache. Returns false if nothing cached.
    bool cancelShell();
    bool hasPreShellCache() const { return _preShellRayModel.has_value(); }

    // Cast a camera ray through (x, y) against the current BRep. On a miss,
    // project onto the view plane through the model centre (perpendicular to
    // the camera look direction). Returns false only when that plane is
    // parallel to the ray.
    bool pickToolPlacement(const vsg::Camera& camera, int32_t x, int32_t y,
                           vsg::dvec3& position, vsg::dvec3& normal) const;

    // Colour applied to subsequently created geometry.
    void setSurfaceColor(const vsg::vec4& color) { _surfaceColor = color; }
    const vsg::vec4& surfaceColor() const { return _surfaceColor; }

    // Scale/centre imported geometry into a unit box at the origin so the
    // existing camera frames it regardless of the model's native units.
    void setFitToUnitBox(bool enable) { _fitToUnitBox = enable; }
    bool fitToUnitBox() const { return _fitToUnitBox; }

    // Active view camera for Ray-GS density / frustum cull. Call after the
    // viewer is framed; noteCameraMoved() restarts the settle debounce.
    void setCamera(vsg::ref_ptr<vsg::Camera> camera);
    void noteCameraMoved();
    // Trackball / scroll events that should kick the Ray-GS view refresh.
    vsg::ref_ptr<vsg::Visitor> createCameraSettleHandler();

    // Once after Vulkan device creation: pick stock / cut-face ray budgets from
    // device-local heap size and CPU cores. No mid-session retune.
    void configureRayBudgets(vsg::ref_ptr<vsg::Device> device);
    std::size_t maxRenderedRays() const { return _maxRenderedRays; }
    std::size_t maxCutFaceRays() const { return _maxCutFaceRays; }

    // Log per-cut timings (boolean, splat patch vs full rebuild) and the
    // interval-pool / splat-buffer occupancy that drives them to stdout.
    void setProfilingEnabled(bool enable) { _profiling = enable; }
    bool profilingEnabled() const { return _profiling; }

    const vsg::ref_ptr<vsg::Group>& scene() const { return _scene; }

private:
    // Draw whatever suits the current view mode.
    void rebuild();

    // After Shell / Cancel shell: refresh Ray line mode and Ray-GS from _rayModel.
    void refreshRayViewsAfterStockEdit();

    // Assemble a StateGroup around one indexed draw. lines selects a line list
    // topology (with culling off) instead of a triangle list. transparent turns
    // on alpha blending and disables depth writes. overlay disables depth so a
    // path on the stock surface stays visible. outDraw receives the draw node.
    vsg::ref_ptr<vsg::Node> buildDrawable(vsg::ref_ptr<vsg::vec3Array> positions,
                                          vsg::ref_ptr<vsg::vec3Array> normals,
                                          vsg::ref_ptr<vsg::vec4Array> colors,
                                          VkVertexInputRate colorRate,
                                          vsg::ref_ptr<vsg::uintArray> indices,
                                          bool lines,
                                          bool transparent = false,
                                          bool overlay = false,
                                          vsg::ref_ptr<vsg::VertexIndexDraw>* outDraw = nullptr) const;

    // Wrap node in the transform that fits bounds into a unit box, if enabled.
    // Mesh and rays share this so both land in the same place.
    vsg::ref_ptr<vsg::Node> applyFit(vsg::ref_ptr<vsg::Node> node,
                                     const BoundingBox& bounds) const;

    // Model-space to world-space transform matching applyFit(), or identity
    // when fitting is off / bounds are empty.
    vsg::dmat4 fitMatrix(const BoundingBox& bounds) const;

    BoundingBox worldStockAabb() const;
    void updateWorldAxesSpecFromStock();

    // Build (or rebuild) the tool subgraph for the current type and the
    // radius/length held in Parameter. preservePose keeps the previous
    // MatrixTransform so a radius edit does not jump the cutter back to the
    // default placement.
    void rebuildTool(bool preservePose);

    // Add / remove the tool transform from the scene without dropping it. The
    // transform stays the pose source while detached, which is how Inspection
    // hides the solid cutter and shows only its wireframe.
    void setToolNodeAttached(bool attached);

    // Model-space radius / length from Parameter, converted into the space the
    // tool is drawn in (world / fitted), matching applyFit()'s scale.
    float worldFromModelLength(double value) const;
    float worldToolRadius() const;
    float worldToolLength() const;

    vsg::ref_ptr<vsg::Node> toolMeshNode(const TriangleMesh& mesh,
                                         const vsg::vec4& color) const;

    // Rebuild the swept-volume scene node from the CPU SweptVolume, when the
    // swept-volume checkbox is on. Inspection draws the current cutter as a
    // wireframe; other modes keep the transparent triangle path.
    void publishSweptVolume();

    void appendToolTrajectory(const vsg::dvec3& position);
    void ensureTrajectoryCapacity();

    // Append one linear sweep step onto the current SweptVolume, if far enough.
    // Accumulate one tip-to-tip segment into the swept volume. Returns true when
    // the sweep mesh changed (so callers can re-run boolean / rebuild Ray-GS).
    bool recordSweepStep(const ToolPose& pose);

    // Accumulate one multi-station sweep (caps at ends only for grinding).
    bool recordSweepPath(const std::vector<ToolPose>& tipPoses);

    // Write the tool matrix from a tip and orthonormal frame, then sweep/boolean.
    void commitToolTip(const vsg::dvec3& tip, const vsg::dvec3& x, const vsg::dvec3& y,
                       const vsg::dvec3& z);

    // Same frame / tip conversion as setToolPose, without committing.
    bool referencePoseToTipFrame(const vsg::dvec3& position, const vsg::dvec3& direction,
                                 vsg::dvec3& tip, vsg::dvec3& x, vsg::dvec3& y,
                                 vsg::dvec3& z) const;

    // Inspection: put the cutter mesh at the current tool pose into _cutSweep
    // and publish it as the swept volume. Returns true when the pose moved
    // far enough (or forceBoolean) that the preview subtract should re-run.
    bool placeInspectionCutter(const ToolPose& pose, bool forceBoolean);

    // Half-width of a Gaussian splat cast along the given axis, in the space
    // applyFit() maps into.
    float splatRadius(const RayModel& rayModel, std::size_t axis) const;

    // Radii for all three axes, already multiplied by the display stride.
    std::array<float, 3> splatRadii(const RayModel& rayModel, int stride) const;

    SplatStyle splatStyle() const;

    // Full Ray-GS rebuild through the Gaussian cache, then attach
    // only when the GPU node is new or its arrays grew.
    void rebuildSplatCache();
    // Global (non-zoom) stride used for stock splat far-face policy.
    int coarseStride() const;
    // Denser stride for orange cut-face mesh only (maxCutFaceRays).
    int cutFaceStride() const;
    void presentSplatCache();
    bool splatOnScreen() const;
    // Inspection: replace-in-window overlay. Does not rewrite _section slots.
    void syncInspectionSectionGrid(const BoundingBox& sectionAabb);

    // Stock AABB ∩ camera frustum AABB in model space (padded). Empty when
    // nothing is on screen; full stock bounds when no camera is set.
    BoundingBox visibleStockAabb() const;
    // View-aware display stride under maxRenderedRays (falls back to global).
    int displayStride() const;
    SplatViewCull splatViewCull() const;
    std::size_t rayCountVisibleAtStride(int stride) const;
    vsg::dmat4 modelToClipMatrix() const;
    void refreshSplatViewForCamera();

    // One --profile line for a completed cut. drawPath names how the display
    // was refreshed: "patch" (splat AABB update), "rebuild" or "no-draw".
    // cloneMs is Inspection's full-stock copy; sectionMs is the cut-face mesh.
    void logCutProfile(double booleanMs, const char* drawPath, double drawMs,
                       const BoundingBox& dirtyModelAabb, double cloneMs = 0.0,
                       double sectionMs = 0.0);

    // Compile a subgraph against the running viewer, then attach it.
    // replaceExisting swaps the model node only; the tool transform is kept.
    void attach(vsg::ref_ptr<vsg::Node> node, bool replaceExisting);

    // Move a resolution to the front of the reuse order, evicting the least
    // recently used models if that puts the cache over its limit.
    void touchRayModel(const Point3d& resolution);

    // Drop every cast model. The rays belong to the BRep they were cast
    // through, so this runs whenever that changes.
    void clearRayModels();

    vsg::ref_ptr<vsgQt::Viewer> _viewer;
    vsg::ref_ptr<vsg::Group> _scene;
    vsg::ref_ptr<vsg::Options> _options;

    // The model currently on display. Replaced wholesale on rebuild; kept
    // separate from the tool so a view-mode change does not drop the cutter.
    vsg::ref_ptr<vsg::Node> _modelNode;

    // Persistent tool placement. Children hold the cutter and optional shank;
    // the matrix moves the tip to the latest pick.
    vsg::ref_ptr<vsg::MatrixTransform> _toolTransform;

    // The last (current) swept volume on the CPU: triangle soup + BVH. Always
    // recorded while a tool is active. The GPU node is only rebuilt when
    // _showSweptVolume is true. Boolean uses _cutSweep (the newest segment
    // only) so the stock can keep prior cuts without re-walking the whole path.
    std::optional<SweptVolume> _sweptVolume;
    std::optional<SweptVolume> _cutSweep;
    vsg::ref_ptr<vsg::Node> _sweptNode;
    bool _showSweptVolume = false;

    // Polyline of tool tips while Operation is not None. LINE_LIST so a None
    // (or retract) can break the strip without connecting the next stroke.
    vsg::ref_ptr<vsg::Node> _trajectoryNode;
    vsg::ref_ptr<vsg::VertexIndexDraw> _trajectoryDraw;
    vsg::ref_ptr<vsg::vec3Array> _trajectoryPositions;
    vsg::ref_ptr<vsg::uintArray> _trajectoryIndices;
    uint32_t _trajectoryPointCount = 0;
    uint32_t _trajectoryIndexCount = 0;
    bool _trajectoryConnect = false;

    // World-origin XYZ gizmo (tube + cone per axis).
    vsg::ref_ptr<vsg::Group> _axesNode;
    WorldAxesSpec _axesSpec{};
    WorldAxisPart _axesSelected = WorldAxisPart::None;

    vsg::vec4 _surfaceColor{0.80f, 0.80f, 0.85f, 1.0f};
    vsg::vec4 _wireframeColor{0.20f, 0.90f, 0.40f, 1.0f};
    vsg::vec4 _toolColor{0.95f, 0.35f, 0.10f, 1.0f};
    vsg::vec4 _shankColor{0.78f, 0.80f, 0.84f, 1.0f};
    vsg::vec4 _sweptColor{0.55f, 0.55f, 0.58f, 0.35f};

    // Rays are coloured by the axis they run along: x red, y green, z blue.
    vsg::vec4 _rayColors[3]{{0.95f, 0.30f, 0.30f, 1.0f},
                            {0.30f, 0.95f, 0.35f, 1.0f},
                            {0.35f, 0.55f, 1.00f, 1.0f}};

    // Albedo of the Gaussian splats: a slightly cool grey, which the shader
    // shades as metal. Kept apart from the wireframe colour so that the two
    // modes can be tuned independently.
    vsg::vec4 _splatColor{0.48f, 0.50f, 0.54f, 1.0f};

    // Peak opacity of a single Gaussian splat. Low enough that overlapping
    // splats visibly mix rather than the nearest one simply covering the rest,
    // high enough that a few of them still build up to a solid surface.
    float _splatOpacity = 0.5f;

    // Incremental GPU splat buffers for Ray-GS. Cleared when the cast model or
    // resolution changes; patched in-place for boolean AABB windows.
    GaussianSplatCache _splatCache;

    bool _fitToUnitBox = true;
    ViewMode _viewMode = ViewMode::Facet;
    ToolType _toolType = ToolType::None;

    // --profile: per-cut timing log, and the cut counter it is keyed by.
    bool _profiling = false;
    long long _cutIndex = 0;

    // Kept so a view mode change can rebuild the geometry from the source
    // topology rather than from whatever is currently in the scene.
    std::optional<BRep> _current;

    // Rays cast through the current BRep, keyed by the resolution used. Point3d
    // is a std::array, so it orders lexicographically and needs no comparator.
    // The keys are matched exactly: resolutions reach here from the spin boxes
    // and from the model bounds, both of which reproduce the same values bit
    // for bit, so an exact match is what "the same setting" means.
    std::map<Point3d, RayModel> _rayModels;

    // Keys in reuse order, least recently used first. A cast model can be tens
    // of megabytes, so only the last few are worth holding on to.
    std::vector<Point3d> _rayModelOrder;
    static constexpr std::size_t maxCachedRayModels = 8;

    // Rays currently held across every cached model, and the ceiling on them.
    // A cast fine enough to be worth keeping as geometry runs to hundreds of
    // megabytes, so counting models is not enough to bound the cache: at the
    // finest resolutions this leaves room for one, and the previous cast is
    // dropped as a new one arrives.
    std::size_t _cachedRays = 0;
    static constexpr std::size_t maxCachedRays = 10000000;

    // How many rays may be handed to the GPU at once. Each one becomes two
    // splat quads. Defaults are balanced mid-range; configureRayBudgets() sets
    // them once from GPU heap / CPU cores after the Vulkan device exists.
    std::size_t _maxRenderedRays = 2000000;
    // Cut-face mesh stride budget (independent of stock splat fill-rate).
    std::size_t _maxCutFaceRays = 4000000;

    // Points into _rayModels (the unmodified cast), or null before anything
    // has been cast.
    const RayModel* _sourceRayModel = nullptr;

    // Result of the accumulated boolean ops. Each new sweep cut uses this as
    // input (falling back to _sourceRayModel for the first cut). _rayModel
    // points here while a subtraction/union session is active.
    std::optional<RayModel> _booleanRayModel;

    // Stock immediately before the last successful Shell. One-slot undo; not
    // keyed into _rayModels (those are resolution-keyed pristine casts).
    std::optional<RayModel> _preShellRayModel;

    // Inspection working copy: cloned from session stock (_booleanRayModel if
    // present, else _sourceRayModel) each preview move, then subtracted.
    // Discarded when leaving Inspection so accumulated cuts in
    // _booleanRayModel stay intact.
    std::optional<RayModel> _inspectionRayModel;

    // Model-space AABB of the last inspection cutter, so the next move can
    // patch both the restored hole and the new one.
    BoundingBox _inspectionPrevAabb;

    // Tip pose last used for an inspection subtract (dead-band).
    std::optional<ToolPose> _inspectionBooleanPose;

    // Last tool tip / axis written by setToolPose, so Inspection can place a
    // cutter when the combo is selected without waiting for another move.
    std::optional<ToolPose> _lastToolPose;

    // What is drawn: _sourceRayModel, &_booleanRayModel, or &_inspectionRayModel.
    const RayModel* _rayModel = nullptr;

    vsg::ref_ptr<vsg::Camera> _camera;
    class QTimer* _splatViewDebounce = nullptr;
    // Short settle so zoom densify (displayStride + matching radii) catches up
    // quickly; lattice coverage no longer depends on maxApparent shrink.
    static constexpr int splatViewDebounceMs = 40;
};

} // namespace app
