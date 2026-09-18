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

// Grinding-wheel 2D cutter: isosceles trapezoid (tip width A, flare over H1 to
// width B) plus a rectangle of width B and height H2. Origin is the shoulder
// rectangle midpoint (CL). +X is the spindle, −Z is the outer rim.
struct GrindingWheelProfile
{
    float tipWidth = 0.01f;
    float shoulderWidth = 0.06f;
    float taperHeight = 0.035f;
    float shoulderHeight = 0.015f;

    float totalHeight() const
    {
        const float h2 = (shoulderHeight > 0.0f) ? shoulderHeight : 0.0f;
        return taperHeight + h2;
    }

    bool valid() const
    {
        return tipWidth > 0.0f && shoulderWidth > 0.0f && taperHeight > 0.0f &&
               shoulderHeight >= 0.0f;
    }
};

// Mill-local closed loop, CCW from tip-left. Origin = shoulder-rectangle
// midpoint. +Z radial (pose.direction), +X spindle. Returns 4 or 6.
inline int grindingWheelLocalLoop(const GrindingWheelProfile& p, vsg::dvec3 out[6])
{
    if (!p.valid()) return 0;
    const double a2 = 0.5 * static_cast<double>(p.tipWidth);
    const double b2 = 0.5 * static_cast<double>(p.shoulderWidth);
    const double h1 = static_cast<double>(p.taperHeight);
    const double h2 = static_cast<double>(p.shoulderHeight);
    const double h2h = 0.5 * h2;
    const double zTip = -(h1 + h2h);
    const double zInner = h2h;
    const double zJunc = -h2h;
    const bool hasRect = h2 > 1.0e-12;
    const bool flared =
        std::abs(static_cast<double>(p.tipWidth) - static_cast<double>(p.shoulderWidth)) >
        1.0e-12;
    if (!hasRect || !flared)
    {
        out[0] = vsg::dvec3(-a2, 0.0, zTip);
        out[1] = vsg::dvec3(a2, 0.0, zTip);
        out[2] = vsg::dvec3(b2, 0.0, zInner);
        out[3] = vsg::dvec3(-b2, 0.0, zInner);
        return 4;
    }
    out[0] = vsg::dvec3(-a2, 0.0, zTip);
    out[1] = vsg::dvec3(a2, 0.0, zTip);
    out[2] = vsg::dvec3(b2, 0.0, zJunc);
    out[3] = vsg::dvec3(b2, 0.0, zInner);
    out[4] = vsg::dvec3(-b2, 0.0, zInner);
    out[5] = vsg::dvec3(-b2, 0.0, zJunc);
    return 6;
}

// Mill frame at a station: +Z radial, +Y feed projected into the plane ⊥ Z
// (profile-plane normal), +X = Y×Z spindle. Plunge along radial keeps the
// preferred-X fallback. Optional prevY flips Y so adjacent loft stations
// do not twist 180°.
inline void grindingWheelMotionFrame(const vsg::dvec3& radial, const vsg::dvec3& along,
                                     vsg::dvec3& x, vsg::dvec3& y, vsg::dvec3& z,
                                     const vsg::dvec3* prevY = nullptr)
{
    const double zLen = vsg::length(radial);
    z = (zLen > 1.0e-12) ? radial / zLen : vsg::dvec3(0.0, 0.0, 1.0);

    auto orthonormalize = [&]() {
        y = vsg::cross(z, x);
        const double yLen = vsg::length(y);
        if (yLen > 1.0e-12) y /= yLen;
        else y = vsg::dvec3(0.0, 1.0, 0.0);
        x = vsg::cross(y, z);
        const double xLen = vsg::length(x);
        if (xLen > 1.0e-12) x /= xLen;
        z = vsg::cross(x, y);
        const double z2 = vsg::length(z);
        if (z2 > 1.0e-12) z /= z2;
    };

    const vsg::dvec3 alongPlanar = along - z * vsg::dot(along, z);
    const double alongLen = vsg::length(alongPlanar);
    // 1e-5: 4-decimal table chords after the 0.1 fit collapse below this and
    // used to snap Y to world +X (wheel swivel on a dense helix).
    constexpr double kPlanarMin = 1.0e-5;
    if (alongLen > kPlanarMin)
    {
        y = alongPlanar / alongLen;
        if (prevY && vsg::dot(y, *prevY) < 0.0) y = -y;
        x = vsg::cross(y, z);
        const double xLen = vsg::length(x);
        if (xLen > 1.0e-12)
        {
            x /= xLen;
            orthonormalize();
            return;
        }
    }

    // Tiny / axial feed: transport the last Y onto this radial.
    if (prevY)
    {
        vsg::dvec3 yHint = *prevY - z * vsg::dot(*prevY, z);
        const double yHintLen = vsg::length(yHint);
        if (yHintLen > 1.0e-12)
        {
            y = yHint / yHintLen;
            x = vsg::cross(y, z);
            const double xLen = vsg::length(x);
            if (xLen > 1.0e-12)
            {
                x /= xLen;
                orthonormalize();
                return;
            }
        }
    }

    vsg::dvec3 prefer(1.0, 0.0, 0.0);
    if (std::abs(vsg::dot(prefer, z)) > 0.95) prefer = vsg::dvec3(0.0, 1.0, 0.0);
    x = prefer - z * vsg::dot(prefer, z);
    const double xLen = vsg::length(x);
    x = (xLen > 1.0e-12) ? x / xLen : vsg::dvec3(1.0, 0.0, 0.0);
    orthonormalize();
    if (prevY && vsg::dot(y, *prevY) < 0.0)
    {
        y = -y;
        x = -x;
    }
}

// Mouse / table location is this far from the tip along +axis: sphere centre
// for ball nose and sphere, fillet-torus centre for bull nose, tip otherwise.
// Grinding wheel: origin is the shoulder-rectangle midpoint (CL reference).
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
// For GrindingWheel, `wheel` is the cutter profile and shankRadius sets the
// spindle offset (axis on the shank edge). radius / height / vertexAngleDeg
// are unused for the cutter body.
TriangleMesh createToolMesh(ToolType type, float radius, float height,
                            int slices = 48, int stacks = 24, int filletStacks = 12,
                            float vertexAngleDeg = 60.0f,
                            float shankRadius = 0.0f,
                            float shankLength = 0.0f,
                            GrindingWheelProfile wheel = {});

// Top of the cutting body along +Z (where a shank cylinder should start).
// For GrindingWheel the CL / shoulder-rectangle mid is at the origin.
inline float toolCuttingTop(ToolType type, float radius, float cuttingLength,
                            float vertexAngleDeg = 60.0f)
{
    if (radius < 0.0f) radius = 0.0f;
    if (cuttingLength < 0.0f) cuttingLength = 0.0f;
    (void)vertexAngleDeg;
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
// cutter (axis on the shank edge). Origin is the shoulder-rectangle midpoint.
TriangleMesh createGrindingShankMesh(float shoulderHeight, float shankRadius, float shankLength,
                                     int slices = 48, int stacks = 8);

} // namespace app
