#include "UcamVsgGuard.h"

#include <QtGui/QExposeEvent>
#include <QtGui/QResizeEvent>

#include <exception>
#include <iostream>
#include <utility>

#include <vulkan/vulkan.h>

namespace app
{
namespace
{

void reportVsgException(const vsg::Exception& e)
{
    std::cerr << "vsg::Exception: " << e.message << " (VkResult " << e.result << ")"
              << std::endl;
    if (e.result == VK_ERROR_DEVICE_LOST)
    {
        if (auto* app = UcamApplication::instance())
            app->stopContinuousUpdate();
    }
}

void reportStdException(const std::exception& e)
{
    std::cerr << "std::exception: " << e.what() << std::endl;
}

template<typename Fn>
void swallow(Fn&& fn)
{
    try
    {
        fn();
    }
    catch (const vsg::Exception& e)
    {
        reportVsgException(e);
    }
    catch (const std::exception& e)
    {
        reportStdException(e);
    }
}

} // namespace

UcamApplication* UcamApplication::instance()
{
    return static_cast<UcamApplication*>(QCoreApplication::instance());
}

void UcamApplication::addViewer(vsg::ref_ptr<vsgQt::Viewer> viewer)
{
    if (!viewer) return;
    for (const auto& existing : _viewers)
    {
        if (existing == viewer) return;
    }
    _viewers.push_back(std::move(viewer));
}

void UcamApplication::stopContinuousUpdate()
{
    for (auto& viewer : _viewers)
    {
        if (!viewer) continue;
        viewer->continuousUpdate = false;
        viewer->requests.store(0);
        viewer->timer.stop();
    }
}

bool UcamApplication::notify(QObject* receiver, QEvent* event)
{
    try
    {
        return QApplication::notify(receiver, event);
    }
    catch (const vsg::Exception& e)
    {
        reportVsgException(e);
        return true;
    }
    catch (const std::exception& e)
    {
        reportStdException(e);
        return true;
    }
}

void SafeViewer::render(double simulationTime)
{
    swallow([&] { vsgQt::Viewer::render(simulationTime); });
}

void SafeVsgWindow::exposeEvent(QExposeEvent* event)
{
    swallow([&] { vsgQt::Window::exposeEvent(event); });
}

void SafeVsgWindow::resizeEvent(QResizeEvent* event)
{
    if (!event || event->size().width() < 1 || event->size().height() < 1)
        return;

    swallow([&] { vsgQt::Window::resizeEvent(event); });
}

} // namespace app
