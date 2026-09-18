// Tool manager dialog: a VSG viewport for browsing cutter geometry.
#pragma once

#include <vsg/all.h>
#include <vsgQt/Viewer.h>
#include <vsgQt/Window.h>

#include <QtWidgets/QDialog>

#include "ToolType.h"

class QCloseEvent;
class QShowEvent;
class QTreeWidget;
class QTreeWidgetItem;

namespace app
{

class ToolManagerDialog : public QDialog
{
    Q_OBJECT

public:
    enum class LeafKind
    {
        Radius,
        CuttingLength,
        ShankLength,
        ShankRadius,
        TipWidth,
        ShoulderWidth,
        TaperHeight,
        ShoulderHeight
    };

    ToolManagerDialog(vsg::ref_ptr<vsg::WindowTraits> sharedTraits,
                      vsg::ref_ptr<vsg::Options> options,
                      int timerInterval,
                      QWidget* parent = nullptr);

    vsg::ref_ptr<vsg::Group> scene() const { return _scene; }
    void previewTool(ToolType type, double radius, double cuttingLength, double shankLength,
                     double shankRadius, double tipWidth = 0.0, double shoulderWidth = 0.0,
                     double taperHeight = 0.0, double shoulderHeight = 0.0);

signals:
    // Fired after a numeric library leaf is committed in this dialog.
    void toolLibraryEntryChanged(int toolType, double radius, double cuttingLength,
                                 double shankLength, double shankRadius, double tipWidth,
                                 double shoulderWidth, double taperHeight,
                                 double shoulderHeight);

protected:
    void showEvent(QShowEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    struct ToolParams
    {
        double radius = 0.05;
        double cuttingLength = 0.14;
        double shankLength = 0.14;
        double shankRadius = 0.06;
        double tipWidth = 0.01;
        double shoulderWidth = 0.06;
        double taperHeight = 0.035;
        double shoulderHeight = 0.015;
    };

    struct PendingPreview
    {
        bool valid = false;
        ToolType type = ToolType::None;
        double radius = 0.0;
        double cuttingLength = 0.0;
        double shankLength = 0.0;
        double shankRadius = 0.0;
        double tipWidth = 0.01;
        double shoulderWidth = 0.06;
        double taperHeight = 0.035;
        double shoulderHeight = 0.015;
    };

    void initializeScene();
    void fillToolLibrary();
    void applyPreview(ToolType type, double radius, double cuttingLength, double shankLength,
                      double shankRadius, double tipWidth, double shoulderWidth,
                      double taperHeight, double shoulderHeight);
    void previewToolItem(QTreeWidgetItem* toolItem);
    void onLibraryItemChanged(QTreeWidgetItem* item, int column);
    void onLibrarySelectionChanged();
    void onLibraryItemDoubleClicked(QTreeWidgetItem* item, int column);
    void selectToolForType(ToolType type);
    ToolParams readToolParams(QTreeWidgetItem* toolItem) const;
    void writeToolParams(QTreeWidgetItem* toolItem, const ToolParams& params);
    QTreeWidgetItem* toolItemFromAny(QTreeWidgetItem* item) const;
    void frameScene();
    void presentPreview(vsg::ref_ptr<vsg::Node> node);

    vsg::ref_ptr<vsg::WindowTraits> _traits;
    vsg::ref_ptr<vsg::Options> _options;
    vsg::ref_ptr<vsgQt::Viewer> _viewer;
    vsg::ref_ptr<vsg::Group> _scene;
    vsg::ref_ptr<vsg::LookAt> _lookAt;
    vsg::ref_ptr<vsg::Perspective> _perspective;
    vsgQt::Window* _window = nullptr;
    QTreeWidget* _toolTree = nullptr;
    PendingPreview _pending;
    int _timerInterval = 8;
    bool _ready = false;
};

} // namespace app
