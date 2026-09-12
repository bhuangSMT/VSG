// ControlCube - CAD view cube drawn in a bottom-left Vulkan inset.
//
// A second vsg::View tracks the main camera's rotation. Dragging the cube
// orbits the main view; a still click on a face, edge tube, or corner sphere
// snaps the trackball (Front is the default -Y look).
#pragma once

#include <cstdint>
#include <optional>

#include <vsg/all.h>

class QWindow;

namespace app
{

class ControlCube : public vsg::Inherit<vsg::Visitor, ControlCube>
{
public:
    static constexpr int insetSize = 165;
    static constexpr int insetMargin = 12;

    ControlCube(vsg::ref_ptr<vsg::Camera> mainCamera,
                vsg::ref_ptr<vsg::Trackball> trackball,
                vsg::ref_ptr<vsg::Options> options,
                vsg::ref_ptr<vsg::Window> window,
                QWindow* qtWindow);

    vsg::ref_ptr<vsg::View> view() const { return _view; }
    vsg::ref_ptr<vsg::ClearAttachments> depthClear() const { return _depthClear; }

    // Window coordinates in the same device-pixel space as VSG pointer events.
    bool contains(int32_t x, int32_t y) const;

    void apply(vsg::FrameEvent& frame) override;
    void apply(vsg::ButtonPressEvent& press) override;
    void apply(vsg::ButtonReleaseEvent& release) override;
    void apply(vsg::MoveEvent& move) override;
    void apply(vsg::ScrollWheelEvent& scroll) override;

private:
    void syncFromMain();
    void updateInset();
    bool unproject(int32_t x, int32_t y, vsg::dvec3& origin, vsg::dvec3& direction) const;
    vsg::dvec3 insetTbc(int32_t x, int32_t y) const;
    void orbitDrag(int32_t x, int32_t y);
    std::optional<vsg::dvec3> pickSnapDirection(int32_t x, int32_t y) const;
    void snapToDirection(const vsg::dvec3& direction);
    void claimPointer(vsg::UIEvent& event);

    vsg::ref_ptr<vsg::Camera> _mainCamera;
    vsg::ref_ptr<vsg::Trackball> _trackball;
    vsg::ref_ptr<vsg::Window> _window;
    QWindow* _qtWindow = nullptr;

    vsg::ref_ptr<vsg::LookAt> _cubeLookAt;
    vsg::ref_ptr<vsg::Orthographic> _cubeOrtho;
    vsg::ref_ptr<vsg::ViewportState> _cubeViewport;
    vsg::ref_ptr<vsg::Camera> _cubeCamera;
    vsg::ref_ptr<vsg::View> _view;
    vsg::ref_ptr<vsg::ClearAttachments> _depthClear;

    int32_t _insetX = 0;
    int32_t _insetY = 0;
    int32_t _insetExtent = insetSize;

    bool _leftPressed = false;
    bool _pressInCube = false;
    bool _dragging = false;
    int32_t _pressX = 0;
    int32_t _pressY = 0;
    int32_t _lastX = 0;
    int32_t _lastY = 0;
    vsg::dvec3 _prevTbc{0.0, 0.0, 1.0};
};

} // namespace app

EVSG_type_name(app::ControlCube);
