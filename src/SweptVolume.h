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

    // Append one linear segment at constant cost.
    // Flat: stadium prism (Minkowski of the cylinder with tipA→tipB).
    // Bull: flat tip stadium + quarter-torus fillet loft + shank stadium prism.
    // Sphere: capsule about the centre path (cylinder + hemispheres).
    // Ball nose: centre-path cylinder, tip + lower motion caps, then shank loft.
    // circleSegments is the azimuthal tessellation for rings / stadium ends.
    void appendSegment(ToolType type,
                       float radius,
                       float height,
                       const ToolPose& tipA,
                       const ToolPose& tipB,
                       int circleSegments = 8);

    // Append triangles from another soup. rebuildHierarchy rebuilds the BVH
    // (needed when this volume is used for boolean). Display-only accumulation
    // can skip it.
    void appendTriangles(const TriangleMesh& extra, bool rebuildHierarchy = true);

private:
    void rebuildBvh();

    TriangleMesh _mesh;
    BVH _bvh;
    std::optional<ToolPose> _lastPose;
};

} // namespace app
