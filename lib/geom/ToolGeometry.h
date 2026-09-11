// ToolGeometry - triangle meshes for the cutters shown under the mouse.
//
// Each mesh sits in a local frame with the tip at the origin and the tool axis
// along +Z (shank pointing the positive way). The caller places that frame with
// a MatrixTransform so the tip meets the surface and +Z follows the tilted
// surface normal.
#pragma once

#include "ToolType.h"
#include "TriangleMesh.h"

namespace app
{

// Corner radius of the bull-nose fillet (same fraction as the display mesh).
inline float bullNoseFilletRadius(float radius)
{
    const float fillet = radius * 0.35f;
    const float cap = radius * 0.999f;
    return (fillet < cap) ? fillet : cap;
}

// Mouse / table location is this far from the tip along +axis: sphere centre
// for ball nose and sphere, fillet-torus centre for bull nose, tip otherwise.
inline float toolCenterOffset(ToolType type, float radius)
{
    if (radius <= 0.0f) return 0.0f;
    switch (type)
    {
    case ToolType::BallNose:
    case ToolType::Sphere:
        return radius;
    case ToolType::BullNose:
        return bullNoseFilletRadius(radius);
    default:
        return 0.0f;
    }
}

// radius is the cutter radius; height is the cylindrical shank length beyond
// the tip geometry. Both are in the same units the mesh will be drawn in.
// slices / stacks / filletStacks control the circular profile tessellation;
// omit them for the high-resolution display mesh.
TriangleMesh createToolMesh(ToolType type, float radius, float height,
                            int slices = 48, int stacks = 24, int filletStacks = 12);

} // namespace app
