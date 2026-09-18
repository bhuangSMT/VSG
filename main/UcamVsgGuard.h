// Guard VSG/Qt integration so swapchain rebuilds and device loss do not abort.
#pragma once

#include <vsg/all.h>
#include <vsgQt/Viewer.h>
#include <vsgQt/Window.h>

#include <QtWidgets/QApplication>

#include <vector>

class QExposeEvent;
class QResizeEvent;

namespace app
{

class UcamApplication : public QApplication
{
public:
    using QApplication::QApplication;

    static UcamApplication* instance();

    void addViewer(vsg::ref_ptr<vsgQt::Viewer> viewer);
    void stopContinuousUpdate();

    bool notify(QObject* receiver, QEvent* event) override;

private:
    std::vector<vsg::ref_ptr<vsgQt::Viewer>> _viewers;
};

class SafeViewer : public vsg::Inherit<vsgQt::Viewer, SafeViewer>
{
public:
    SafeViewer(int msecTimerInterval = 0) :
        Inherit(msecTimerInterval)
    {
    }

    void render(double simulationTime = vsg::Viewer::UseTimeSinceStartPoint) override;
};

class SafeVsgWindow : public vsgQt::Window
{
public:
    using vsgQt::Window::Window;

protected:
    void exposeEvent(QExposeEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
};

} // namespace app

EVSG_type_name(app::SafeViewer);
