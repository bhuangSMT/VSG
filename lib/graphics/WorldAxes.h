// WorldAxes - RGB XYZ gizmo at the world origin (tube shaft + cone tip).
//
// Tubes and cones are separately pickable; the active part is highlighted.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <vsg/maths/vec3.h>
#include <vsg/maths/vec4.h>

#include "TriangleMesh.h"

namespace app
{

enum class WorldAxisPart : int
{
    None = -1,
    XTube = 0,
    XCone,
    YTube,
    YCone,
    ZTube,
    ZCone
};

inline int worldAxisIndex(WorldAxisPart part)
{
    if (part == WorldAxisPart::None) return -1;
    return static_cast<int>(part) / 2;
}

inline bool worldAxisIsCone(WorldAxisPart part)
{
    const int i = static_cast<int>(part);
    return i >= 0 && (i % 2) == 1;
}

struct WorldAxesSpec
{
    // Fallback sizes before a stock AABB is known. RenderManager replaces these
    // with length = 0.5 * stockDiagonal and radii proportional to that length.
    float length = 1.0f;
    float tubeRadius = 0.018f;
    float coneRadius = 0.040f;
    float coneLength = 0.18f;
};

// Base RGB for X / Y / Z. Selected parts use the bright variants.
vsg::vec4 worldAxisColor(int axisIndex, bool selected);

// Six meshes: X tube, X cone, Y tube, Y cone, Z tube, Z cone.
std::array<TriangleMesh, 6> buildWorldAxesMeshes(const WorldAxesSpec& spec,
                                                 WorldAxisPart selected);

// Closest tube/cone hit for origin + t * direction.
std::optional<WorldAxisPart> pickWorldAxes(const WorldAxesSpec& spec,
                                           const vsg::dvec3& origin,
                                           const vsg::dvec3& direction);

} // namespace app
