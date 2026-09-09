// SweptVolume - the cutter path recorded on the CPU.
//
// Holds the triangle soup of the last (current) sweep together with a BVH over
// those triangles. Each linear tool step appends geometry and rebuilds the BVH
// so the hierarchy stays in step with the mesh. RenderManager keeps at most
// one of these; clearing the sweep drops it.
#pragma once

#include <optional>

#include <vsg/maths/vec3.h>

#include "BVH.h"
#include "ToolType.h"
#include "TriangleMesh.h"

namespace app
{

struct ToolPose
{
    vsg::dvec3 position{0.0, 0.0, 0.0};
    vsg::dvec3 direction{0.0, 0.0, 1.0};
};

class SweptVolume
{
public:
    bool empty() const { return _mesh.triangles.empty(); }

    const TriangleMesh& mesh() const { return _mesh; }
    const BVH& bvh() const { return _bvh; }

    const std::optional<ToolPose>& lastPose() const { return _lastPose; }
    void setLastPose(ToolPose pose) { _lastPose = std::move(pose); }
    void clearLastPose() { _lastPose.reset(); }

    // Drop mesh and BVH but keep the remembered tip pose, so the next segment
    // can still connect from the last tip without reseeding.
    void clearGeometry();

    // Drop mesh, BVH, and the remembered tip pose.
    void clear();

    // Append one linear segment at constant cost:
    //  1) place the tool's meridional profile at tipA and tipB (rings around
    //     the tool axis) and loft the matching rings together;
    //  2) close each tip by revolving that profile 180° around the tool axis
    //     (frame.z), with flipWinding selecting revolve sense and winding.
    // circleSegments is the azimuthal tessellation of the loft rings.
    void appendSegment(ToolType type,
                       float radius,
                       float height,
                       const ToolPose& tipA,
                       const ToolPose& tipB,
                       int circleSegments = 8);

private:
    void rebuildBvh();

    TriangleMesh _mesh;
    BVH _bvh;
    std::optional<ToolPose> _lastPose;
};

} // namespace app
