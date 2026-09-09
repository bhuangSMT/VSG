// ToolTracker - moves the active cutter with the mouse.
//
// On each move it casts a ray through the cursor against the current BRep, falls
// back to the view projection plane through the model centre on a miss, then
// tilts the hit normal by 15 degrees toward the camera and hands the pose to
// the RenderManager.
#pragma once

#include <cmath>
#include <memory>

#include <vsg/all.h>

#include "Parameter.h"
#include "RenderManager.h"

namespace app
{

class ToolTracker : public vsg::Inherit<vsg::Visitor, ToolTracker>
{
public:
    ToolTracker(std::shared_ptr<RenderManager> renderManager, vsg::ref_ptr<vsg::Camera> camera) :
        _renderManager(std::move(renderManager)),
        _camera(camera)
    {
    }

    void apply(vsg::MoveEvent& move) override
    {
        update(move.x, move.y);
    }

    void apply(vsg::ButtonPressEvent& press) override
    {
        update(press.x, press.y);
    }

private:
    void update(int32_t x, int32_t y)
    {
        if (!_renderManager || !_camera) return;
        if (Parameter::instance().toolType() == ToolType::None) return;

        vsg::dvec3 position;
        vsg::dvec3 normal;
        if (!_renderManager->pickToolPlacement(*_camera, x, y, position, normal)) return;

        const vsg::dvec3 direction = tiltTowardCamera(position, normal);
        _renderManager->setToolPose(position, direction);
    }

    vsg::dvec3 tiltTowardCamera(const vsg::dvec3& position, const vsg::dvec3& normal) const
    {
        constexpr double tiltRadians = 15.0 * (3.14159265358979323846 / 180.0);

        vsg::dvec3 n = vsg::normalize(normal);

        // Eye position from the view matrix; tilt the normal toward the view so
        // the tool leans slightly to face the camera.
        const vsg::dmat4 view = _camera->viewMatrix->transform();
        const vsg::dmat4 eyeToWorld = vsg::inverse(view);
        const vsg::dvec3 eye(eyeToWorld(3, 0), eyeToWorld(3, 1), eyeToWorld(3, 2));

        vsg::dvec3 toEye = eye - position;
        const double toEyeLen = vsg::length(toEye);
        if (toEyeLen <= 0.0) return n;
        toEye /= toEyeLen;

        vsg::dvec3 axis = vsg::cross(n, toEye);
        const double axisLen = vsg::length(axis);
        if (axisLen < 1.0e-8)
        {
            // Normal already faces the camera (or away); pick any perpendicular.
            axis = (std::abs(n.z) < 0.9) ? vsg::cross(n, vsg::dvec3(0.0, 0.0, 1.0))
                                         : vsg::cross(n, vsg::dvec3(1.0, 0.0, 0.0));
            const double len = vsg::length(axis);
            if (len < 1.0e-8) return n;
            axis /= len;
        }
        else
        {
            axis /= axisLen;
        }

        const double cosA = std::cos(tiltRadians);
        const double sinA = std::sin(tiltRadians);
        // Rodrigues rotation of n about axis by tiltRadians.
        return vsg::normalize(n * cosA + vsg::cross(axis, n) * sinA +
                              axis * vsg::dot(axis, n) * (1.0 - cosA));
    }

    std::shared_ptr<RenderManager> _renderManager;
    vsg::ref_ptr<vsg::Camera> _camera;
};

} // namespace app
