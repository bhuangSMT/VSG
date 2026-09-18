// ToolPoseLog - pending Interactive tool samples waiting for the table to drain.
//
// Mouse moves push into pending. A coalesced queued event then swaps the vector
// empty and fills the widget on the Qt GUI thread.
#pragma once

#include <cmath>
#include <vector>

#include <vsg/maths/vec3.h>

#include "SweptVolume.h"

namespace app
{

struct ToolSample
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    bool hasFeed = false;
    vsg::dvec3 feed{0.0, 0.0, 0.0};
};

// Same tool frame as RenderManager::setToolPose, then XYZ Tait-Bryan in degrees
// (A about X, B about Y, C about Z). C comes from the constructed X axis.
inline ToolSample toolSampleFromPose(const ToolPose& pose)
{
    vsg::dvec3 z = pose.direction;
    const double zLen = vsg::length(z);
    if (zLen <= 0.0) z = vsg::dvec3(0.0, 0.0, 1.0);
    else z /= zLen;

    vsg::dvec3 up(0.0, 0.0, 1.0);
    if (std::abs(vsg::dot(z, up)) > 0.95) up = vsg::dvec3(1.0, 0.0, 0.0);

    vsg::dvec3 x = vsg::cross(up, z);
    const double xLen = vsg::length(x);
    if (xLen <= 0.0) x = vsg::dvec3(1.0, 0.0, 0.0);
    else x /= xLen;

    const vsg::dvec3 y = vsg::cross(z, x);

    constexpr double radToDeg = 180.0 / 3.14159265358979323846;
    const double xz = x.z;
    const double clamped = (xz < -1.0) ? -1.0 : ((xz > 1.0) ? 1.0 : xz);
    const double b = std::asin(-clamped);
    const double a = std::atan2(y.z, z.z);
    const double c = std::atan2(x.y, x.x);

    ToolSample sample;
    sample.x = pose.position.x;
    sample.y = pose.position.y;
    sample.z = pose.position.z;
    sample.a = a * radToDeg;
    sample.b = b * radToDeg;
    sample.c = c * radToDeg;
    return sample;
}

// Inverse of toolSampleFromPose: tip at XYZ, tool axis from A,B,C degrees
// (R = Rz(C) * Ry(B) * Rx(A)).
inline ToolPose toolPoseFromSample(const ToolSample& sample)
{
    constexpr double degToRad = 3.14159265358979323846 / 180.0;
    const double a = sample.a * degToRad;
    const double b = sample.b * degToRad;
    const double c = sample.c * degToRad;
    const double ca = std::cos(a);
    const double sa = std::sin(a);
    const double cb = std::cos(b);
    const double sb = std::sin(b);
    const double cc = std::cos(c);
    const double sc = std::sin(c);

    vsg::dvec3 z(cc * sb * ca + sc * sa, sc * sb * ca - cc * sa, cb * ca);
    const double zLen = vsg::length(z);
    if (zLen > 0.0) z /= zLen;
    else z = vsg::dvec3(0.0, 0.0, 1.0);

    return ToolPose{vsg::dvec3(sample.x, sample.y, sample.z), z};
}

class ToolPoseLog
{
public:
    void push(const ToolSample& sample) { pending.push_back(sample); }

    std::vector<ToolSample> take()
    {
        std::vector<ToolSample> out;
        out.swap(pending);
        drainQueued = false;
        return out;
    }

    std::vector<ToolSample> pending;
    bool drainQueued = false;
};

} // namespace app
