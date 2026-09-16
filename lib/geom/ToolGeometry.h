// ToolGeometry - triangle meshes for the cutters shown under the mouse.
//
// Each mesh sits in a local frame with the tip at the origin and the tool axis
// along +Z (shank pointing the positive way). The caller places that frame with
// a MatrixTransform so the tip meets the surface and +Z follows the tilted
// surface normal.
#pragma once

#include <cmath>

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

// Half-length of the grinding-wheel triangle base (from centre). Altitude H,
// tip angle vertexAngleDeg; base lies parallel to the horizontal spindle.
inline float grindingWheelHalfBase(float height, float vertexAngleDeg)
{
    if (!(height > 0.0f)) return 0.0f;
    const float halfRad = vertexAngleDeg * 0.017453292519943295f * 0.5f; // deg → rad / 2
    if (!(halfRad > 0.0f) || halfRad >= 1.5707963267948966f) return 0.0f;
    return height * std::tan(halfRad);
}

// Mouse / table location is this far from the tip along +axis: sphere centre
// for ball nose and sphere, fillet-torus centre for bull nose, tip otherwise.
// Grinding wheel: origin is the triangle-base midpoint (CL reference).
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
    case ToolType::GrindingWheel:
        return 0.0f;
    default:
        return 0.0f;
    }
}

// radius is the cutter radius; height is the cylindrical cutting length beyond
// the tip geometry. Both are in the same units the mesh will be drawn in.
// For GrindingWheel, radius is altitude H and vertexAngleDeg is the tip angle;
// shankRadius sets the spindle offset (axis on the shank edge). height /
// shankLength are unused for the cutter body.
TriangleMesh createToolMesh(ToolType type, float radius, float height,
                            int slices = 48, int stacks = 24, int filletStacks = 12,
                            float vertexAngleDeg = 60.0f,
                            float shankRadius = 0.0f,
                            float shankLength = 0.0f);

// Top of the cutting body along +Z (where a shank cylinder should start).
// For GrindingWheel the CL / base mid is at the origin.
inline float toolCuttingTop(ToolType type, float radius, float cuttingLength,
                            float vertexAngleDeg = 60.0f)
{
    if (radius < 0.0f) radius = 0.0f;
    if (cuttingLength < 0.0f) cuttingLength = 0.0f;
    switch (type)
    {
    case ToolType::Sphere:
        return 2.0f * radius;
    case ToolType::BallNose:
        return radius + cuttingLength;
    case ToolType::BullNose:
        return bullNoseFilletRadius(radius) + cuttingLength;
    case ToolType::FlatNose:
        return cuttingLength;
    case ToolType::GrindingWheel:
        return 0.0f;
    case ToolType::None:
        break;
    }
    return cuttingLength;
}

// Cylindrical shank sitting on the cutting body at z0, along +Z.
TriangleMesh createShankMesh(float radius, float z0, float height, int slices = 48);

// Grinding-wheel shank: rectangle revolved about the same +X spindle as the
// cutter (axis on the shank edge). Origin is the triangle-base midpoint.
TriangleMesh createGrindingShankMesh(float wheelHeight, float shankRadius, float shankLength,
                                     int slices = 48, int stacks = 8);

} // namespace app
