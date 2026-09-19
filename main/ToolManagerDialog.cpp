#include "ToolManagerDialog.h"
#include "UcamVsgGuard.h"

#include <algorithm>

#include <QtGui/QCloseEvent>
#include <QtGui/QColor>
#include <QtGui/QFont>
#include <QtGui/QKeySequence>
#include <QtGui/QShortcut>
#include <QtGui/QShowEvent>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QApplication>
#include <QtWidgets/QColorDialog>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSplitter>
#include <QtWidgets/QTreeWidget>
#include <QtWidgets/QTreeWidgetItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

#include <vsg/utils/Builder.h>

#include "Parameter.h"
#include "ToolGeometry.h"
#include "TriangleMesh.h"

namespace app
{
namespace
{

constexpr int kRoleType = Qt::UserRole;
constexpr int kRoleLeafKind = Qt::UserRole + 1;
constexpr int kRoleValue = Qt::UserRole + 2;
constexpr int kRoleId = Qt::UserRole + 3;
constexpr int kRoleColor = Qt::UserRole + 4;

const QColor kDefaultCutQColor = QColor::fromRgbF(0.95, 0.35, 0.10);
const vsg::vec4 kMetalColor{0.78f, 0.80f, 0.84f, 1.0f};

vsg::vec4 cutColorVec(const QColor& color)
{
    const QColor c = color.isValid() ? color : kDefaultCutQColor;
    return {static_cast<float>(c.redF()), static_cast<float>(c.greenF()),
            static_cast<float>(c.blueF()), 1.0f};
}

bool parsePositive(const QString& text, double* out)
{
    bool ok = false;
    const double value = text.trimmed().toDouble(&ok);
    if (!ok || !(value > 0.0)) return false;
    *out = value;
    return true;
}

bool parseNonNegative(const QString& text, double* out)
{
    bool ok = false;
    const double value = text.trimmed().toDouble(&ok);
    if (!ok || value < 0.0) return false;
    *out = value;
    return true;
}

const char* toolTypeLabel(ToolType type)
{
    switch (type)
    {
    case ToolType::BullNose: return "Bull nose";
    case ToolType::FlatNose: return "Flat nose";
    case ToolType::BallNose: return "Ball nose";
    case ToolType::Sphere: return "Sphere";
    case ToolType::GrindingWheel: return "Grinding wheel";
    case ToolType::None: return "None";
    }
    return "None";
}

const char* leafLabel(ToolManagerDialog::LeafKind kind)
{
    switch (kind)
    {
    case ToolManagerDialog::LeafKind::Radius: return "Radius";
    case ToolManagerDialog::LeafKind::CuttingLength: return "Cutting length";
    case ToolManagerDialog::LeafKind::ShankLength: return "Shank length";
    case ToolManagerDialog::LeafKind::ShankRadius: return "Shank radius";
    case ToolManagerDialog::LeafKind::TipWidth: return "Tip width";
    case ToolManagerDialog::LeafKind::ShoulderWidth: return "Shoulder width";
    case ToolManagerDialog::LeafKind::TaperHeight: return "Taper height";
    case ToolManagerDialog::LeafKind::ShoulderHeight: return "Shoulder height";
    }
    return "Value";
}

QString leafText(ToolManagerDialog::LeafKind kind, double value)
{
    return QString("%1: %2").arg(QString::fromLatin1(leafLabel(kind))).arg(value, 0, 'f', 6);
}

QTreeWidgetItem* makeLeaf(ToolManagerDialog::LeafKind kind, double value)
{
    auto* leaf = new QTreeWidgetItem();
    leaf->setFlags(leaf->flags() | Qt::ItemIsEditable | Qt::ItemIsEnabled | Qt::ItemIsSelectable);
    leaf->setData(0, kRoleLeafKind, static_cast<int>(kind));
    leaf->setData(0, kRoleValue, value);
    leaf->setText(0, leafText(kind, value));
    return leaf;
}

void setLeafValue(QTreeWidgetItem* leaf, double value)
{
    if (!leaf) return;
    const auto kind =
        static_cast<ToolManagerDialog::LeafKind>(leaf->data(0, kRoleLeafKind).toInt());
    leaf->setData(0, kRoleValue, value);
    leaf->setText(0, leafText(kind, value));
}

vsg::ref_ptr<vsg::Node> makePlaceholder(vsg::ref_ptr<vsg::Options> options)
{
    auto builder = vsg::Builder::create();
    builder->options = options;
    vsg::GeometryInfo box;
    box.dx.set(1.0f, 0.0f, 0.0f);
    box.dy.set(0.0f, 1.0f, 0.0f);
    box.dz.set(0.0f, 0.0f, 1.0f);
    box.color.set(0.55f, 0.62f, 0.75f, 1.0f);
    return builder->createBox(box, vsg::StateInfo{});
}

vsg::ref_ptr<vsg::Node> makeToolNode(const TriangleMesh& mesh,
                                     vsg::ref_ptr<vsg::Options> options,
                                     const vsg::vec4& color,
                                     bool metal)
{
    if (mesh.triangles.empty()) return {};

    const auto vertexCount = mesh.triangles.size() * 3;
    auto positions = vsg::vec3Array::create(vertexCount);
    auto normals = vsg::vec3Array::create(vertexCount);
    auto colors = vsg::vec4Array::create(vertexCount, color);
    auto texcoords = vsg::vec2Array::create(vertexCount, vsg::vec2(0.0f, 0.0f));
    auto indices = vsg::uintArray::create(vertexCount);

    std::uint32_t i = 0;
    for (const MeshTriangle& tri : mesh.triangles)
    {
        (*positions)[i] = tri.v0;
        (*positions)[i + 1] = tri.v1;
        (*positions)[i + 2] = tri.v2;
        (*normals)[i] = tri.normal;
        (*normals)[i + 1] = tri.normal;
        (*normals)[i + 2] = tri.normal;
        (*indices)[i] = i;
        (*indices)[i + 1] = i + 1;
        (*indices)[i + 2] = i + 2;
        i += 3;
    }

    auto shaderSet = vsg::createPhongShaderSet(options);
    auto gpc = vsg::GraphicsPipelineConfigurator::create(shaderSet);
    if (const auto& materialBinding = shaderSet->getDescriptorBinding("material"))
    {
        auto material = vsg::PhongMaterialValue::create();
        auto& phong = material->value();
        phong.ambient = color;
        phong.diffuse = color;
        if (metal)
        {
            phong.specular = vsg::vec4(0.95f, 0.95f, 0.97f, 1.0f);
            phong.shininess = 80.0f;
        }
        gpc->assignDescriptor("material", material);
    }
    gpc->enableArray("vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, 8);
    gpc->enableArray("vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, 16);

    auto draw = vsg::VertexIndexDraw::create();
    draw->assignArrays(vsg::DataList{positions, normals, texcoords, colors});
    draw->assignIndices(indices);
    draw->indexCount = static_cast<std::uint32_t>(vertexCount);
    draw->instanceCount = 1;

    gpc->init();
    vsg::StateCommands stateCommands;
    gpc->copyTo(stateCommands);

    auto stateGroup = vsg::StateGroup::create();
    stateGroup->stateCommands.swap(stateCommands);
    stateGroup->prototypeArrayState = gpc->getSuitableArrayState();
    stateGroup->addChild(draw);
    return stateGroup;
}

} // namespace

ToolManagerDialog::ToolManagerDialog(vsg::ref_ptr<vsg::WindowTraits> sharedTraits,
                                     vsg::ref_ptr<vsg::Options> options,
                                     int timerInterval,
                                     QWidget* parent) :
    QDialog(parent),
    _options(options),
    _timerInterval(timerInterval)
{
    setWindowTitle("Tool manager");
    setModal(false);
    resize(1100, 640);

    _traits = vsg::WindowTraits::create();
    _traits->windowTitle = "Tool manager";
    _traits->width = 640;
    _traits->height = 480;
    if (sharedTraits)
    {
        _traits->device = sharedTraits->device;
        _traits->samples = sharedTraits->samples;
    }

    _scene = vsg::Group::create();
    _viewer = SafeViewer::create();
    if (auto* app = UcamApplication::instance())
        app->addViewer(_viewer);
    _window = new SafeVsgWindow(_viewer, _traits);

    auto* viewport = QWidget::createWindowContainer(_window, this);
    viewport->setMinimumSize(320, 240);
    viewport->setFocusPolicy(Qt::StrongFocus);

    auto* left = new QWidget();
    auto* leftLayout = new QVBoxLayout(left);
    leftLayout->setContentsMargins(0, 0, 0, 0);
    leftLayout->setSpacing(8);

    auto* libraryTitle = new QLabel("Tool library");
    QFont titleFont = libraryTitle->font();
    titleFont.setBold(true);
    libraryTitle->setFont(titleFont);
    leftLayout->addWidget(libraryTitle);

    auto* buttons = new QWidget();
    auto* buttonRow = new QHBoxLayout(buttons);
    buttonRow->setContentsMargins(0, 0, 0, 0);
    buttonRow->setSpacing(6);
    _newToolButton = new QPushButton("New tool");
    _deleteButton = new QPushButton("Delete");
    _colorButton = new QPushButton("Color");
    buttonRow->addWidget(_newToolButton);
    buttonRow->addWidget(_deleteButton);
    buttonRow->addWidget(_colorButton);
    buttonRow->addStretch(1);
    leftLayout->addWidget(buttons);

    _toolTree = new QTreeWidget();
    _toolTree->setHeaderLabels({QStringLiteral("Id"), QStringLiteral("Type")});
    _toolTree->setRootIsDecorated(true);
    _toolTree->setUniformRowHeights(false);
    _toolTree->setSelectionMode(QAbstractItemView::SingleSelection);
    _toolTree->setEditTriggers(QAbstractItemView::DoubleClicked |
                               QAbstractItemView::EditKeyPressed |
                               QAbstractItemView::AnyKeyPressed);
    _toolTree->setColumnCount(2);
    _toolTree->setColumnWidth(0, 56);
    _toolTree->header()->setStretchLastSection(true);
    _toolTree->setFocusPolicy(Qt::StrongFocus);
    leftLayout->addWidget(_toolTree, 1);

    fillToolLibrary();

    auto* splitter = new QSplitter();
    splitter->addWidget(left);
    splitter->addWidget(viewport);
    splitter->setStretchFactor(0, 0);
    splitter->setStretchFactor(1, 1);
    splitter->setSizes({320, 780});

    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(8, 8, 8, 8);
    root->addWidget(splitter);

    connect(_toolTree, &QTreeWidget::itemChanged, this, &ToolManagerDialog::onLibraryItemChanged);
    connect(_toolTree, &QTreeWidget::itemSelectionChanged, this,
            &ToolManagerDialog::onLibrarySelectionChanged);
    connect(_toolTree, &QTreeWidget::itemDoubleClicked, this,
            &ToolManagerDialog::onLibraryItemDoubleClicked);
    connect(_newToolButton, &QPushButton::clicked, this, &ToolManagerDialog::onNewTool);
    connect(_deleteButton, &QPushButton::clicked, this, &ToolManagerDialog::deleteSelectedTool);
    connect(_colorButton, &QPushButton::clicked, this, &ToolManagerDialog::onPickColor);

    auto* deleteShortcut = new QShortcut(QKeySequence::Delete, _toolTree);
    deleteShortcut->setContext(Qt::WidgetShortcut);
    connect(deleteShortcut, &QShortcut::activated, this, &ToolManagerDialog::deleteSelectedTool);
    auto* deleteKeyShortcut = new QShortcut(QKeySequence(Qt::Key_Delete), _toolTree);
    deleteKeyShortcut->setContext(Qt::WidgetShortcut);
    connect(deleteKeyShortcut, &QShortcut::activated, this, &ToolManagerDialog::deleteSelectedTool);
    auto* backspaceShortcut = new QShortcut(QKeySequence(Qt::Key_Backspace), _toolTree);
    backspaceShortcut->setContext(Qt::WidgetShortcut);
    connect(backspaceShortcut, &QShortcut::activated, this, &ToolManagerDialog::deleteSelectedTool);

    updateLibraryActions();
}

void ToolManagerDialog::showEvent(QShowEvent* event)
{
    QDialog::showEvent(event);
    initializeScene();
}

void ToolManagerDialog::closeEvent(QCloseEvent* event)
{
    hide();
    event->ignore();
}

void ToolManagerDialog::previewTool(int toolId, ToolType type, double radius, double cuttingLength,
                                    double shankLength, double shankRadius, double tipWidth,
                                    double shoulderWidth, double taperHeight,
                                    double shoulderHeight)
{
    if (toolId > 0) selectToolForId(toolId);
    else selectToolForType(type);
    QTreeWidgetItem* toolItem = nullptr;
    if (_toolTree)
    {
        for (int i = 0; i < _toolTree->topLevelItemCount(); ++i)
        {
            QTreeWidgetItem* item = _toolTree->topLevelItem(i);
            if (!item) continue;
            if (toolId > 0 && item->data(0, kRoleId).toInt() == toolId)
            {
                toolItem = item;
                break;
            }
            if (toolId <= 0 && item->data(0, kRoleType).toInt() == static_cast<int>(type))
            {
                toolItem = item;
                break;
            }
        }
    }
    if (toolItem)
    {
        ToolParams params = readToolParams(toolItem);
        if (radius > 0.0) params.radius = radius;
        if (cuttingLength > 0.0) params.cuttingLength = cuttingLength;
        if (shankLength > 0.0) params.shankLength = shankLength;
        if (shankRadius > 0.0) params.shankRadius = shankRadius;
        const bool haveWheel = tipWidth > 0.0 || shoulderWidth > 0.0 || taperHeight > 0.0;
        if (haveWheel)
        {
            if (tipWidth > 0.0) params.tipWidth = tipWidth;
            if (shoulderWidth > 0.0) params.shoulderWidth = shoulderWidth;
            if (taperHeight > 0.0) params.taperHeight = taperHeight;
            params.shoulderHeight = std::max(0.0, shoulderHeight);
        }
        if (type == ToolType::GrindingWheel)
        {
            params.cuttingLength = params.taperHeight + params.shoulderHeight;
            params.radius = params.cuttingLength;
        }
        writeToolParams(toolItem, params);
    }
    const QColor color = toolColor(toolItem);
    if (!_ready)
    {
        _pending = {true, type, radius, cuttingLength, shankLength, shankRadius,
                    tipWidth, shoulderWidth, taperHeight, shoulderHeight, color};
        return;
    }
    applyPreview(type, radius, cuttingLength, shankLength, shankRadius, tipWidth, shoulderWidth,
                 taperHeight, shoulderHeight, color);
}

ToolManagerDialog::ToolParams ToolManagerDialog::defaultLibraryParams(ToolType type) const
{
    const Parameter& store = Parameter::instance();
    ToolParams params;
    params.radius = store.toolRadius() > 0.0 ? store.toolRadius() : 0.05;
    params.cuttingLength = store.toolLength() > 0.0 ? store.toolLength() : params.radius * 2.8;
    params.shankLength = params.cuttingLength;
    params.shankRadius =
        (type == ToolType::Sphere) ? params.radius * 0.6 : params.radius * 1.2;
    params.tipWidth = store.toolWheelTipWidth() > 0.0 ? store.toolWheelTipWidth() : 0.01;
    params.shoulderWidth =
        store.toolWheelShoulderWidth() > 0.0 ? store.toolWheelShoulderWidth() : 0.06;
    params.taperHeight =
        store.toolWheelTaperHeight() > 0.0 ? store.toolWheelTaperHeight() : 0.035;
    params.shoulderHeight =
        store.toolWheelShoulderHeight() >= 0.0 ? store.toolWheelShoulderHeight() : 0.015;
    if (type == ToolType::GrindingWheel)
    {
        params.cuttingLength = params.taperHeight + params.shoulderHeight;
        params.radius = params.cuttingLength;
    }
    return params;
}

int ToolManagerDialog::nextToolId() const
{
    int next = 1;
    if (!_toolTree) return next;
    for (int i = 0; i < _toolTree->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem* item = _toolTree->topLevelItem(i);
        if (!item) continue;
        next = std::max(next, item->data(0, kRoleId).toInt() + 1);
    }
    return next;
}

QColor ToolManagerDialog::toolColor(QTreeWidgetItem* toolItem) const
{
    if (!toolItem) return kDefaultCutQColor;
    const QColor color = toolItem->data(0, kRoleColor).value<QColor>();
    return color.isValid() ? color : kDefaultCutQColor;
}

QTreeWidgetItem* ToolManagerDialog::currentToolItem() const
{
    if (!_toolTree) return nullptr;
    return toolItemFromAny(_toolTree->currentItem());
}

void ToolManagerDialog::fillToolLibrary()
{
    if (!_toolTree) return;

    struct Seed
    {
        int id;
        ToolType type;
    };
    const Seed seeds[] = {
        {1, ToolType::BullNose},     {2, ToolType::FlatNose},
        {3, ToolType::BallNose},     {4, ToolType::Sphere},
        {5, ToolType::GrindingWheel},
    };

    const bool blocked = _toolTree->blockSignals(true);
    _toolTree->clear();
    for (const Seed& seed : seeds)
        addToolRow(seed.id, seed.type, defaultLibraryParams(seed.type), kDefaultCutQColor);
    _toolTree->blockSignals(blocked);
}

QTreeWidgetItem* ToolManagerDialog::addToolRow(int id, ToolType type, const ToolParams& params,
                                               const QColor& color)
{
    if (!_toolTree) return nullptr;

    auto* tool = new QTreeWidgetItem();
    tool->setFlags((tool->flags() | Qt::ItemIsEnabled | Qt::ItemIsSelectable) &
                   ~Qt::ItemIsEditable);
    tool->setData(0, kRoleId, id);
    tool->setData(0, kRoleType, static_cast<int>(type));
    tool->setData(0, kRoleColor, color.isValid() ? color : kDefaultCutQColor);
    tool->setText(0, QString::number(id));
    _toolTree->addTopLevelItem(tool);
    attachTypeCombo(tool);
    rebuildLeaves(tool, &params);
    tool->setExpanded(false);
    return tool;
}

void ToolManagerDialog::attachTypeCombo(QTreeWidgetItem* toolItem)
{
    if (!toolItem || !_toolTree) return;

    auto* combo = new QComboBox(_toolTree);
    const ToolType types[] = {
        ToolType::BullNose, ToolType::FlatNose, ToolType::BallNose,
        ToolType::Sphere, ToolType::GrindingWheel,
    };
    for (ToolType type : types)
        combo->addItem(QString::fromLatin1(toolTypeLabel(type)), static_cast<int>(type));
    const int index = combo->findData(toolItem->data(0, kRoleType));
    if (index >= 0) combo->setCurrentIndex(index);
    combo->setFocusPolicy(Qt::StrongFocus);
    connect(combo, &QComboBox::currentIndexChanged, this, [this, toolItem](int) {
        onToolTypeChanged(toolItem);
    });
    _toolTree->setItemWidget(toolItem, 1, combo);
}

void ToolManagerDialog::rebuildLeaves(QTreeWidgetItem* toolItem, const ToolParams* overrideParams)
{
    if (!toolItem || !_toolTree) return;

    ToolParams params = overrideParams ? *overrideParams : readToolParams(toolItem);
    const auto type = static_cast<ToolType>(toolItem->data(0, kRoleType).toInt());
    if (type == ToolType::GrindingWheel)
    {
        params.cuttingLength = params.taperHeight + params.shoulderHeight;
        params.radius = params.cuttingLength;
    }

    const bool blocked = _toolTree->blockSignals(true);
    while (toolItem->childCount() > 0)
        delete toolItem->takeChild(0);

    auto addLeaf = [&](LeafKind kind, double value) {
        toolItem->addChild(makeLeaf(kind, value));
        if (QTreeWidgetItem* leaf = toolItem->child(toolItem->childCount() - 1))
            leaf->setFirstColumnSpanned(true);
    };

    if (type == ToolType::GrindingWheel)
    {
        addLeaf(LeafKind::TipWidth, params.tipWidth);
        addLeaf(LeafKind::ShoulderWidth, params.shoulderWidth);
        addLeaf(LeafKind::TaperHeight, params.taperHeight);
        addLeaf(LeafKind::ShoulderHeight, params.shoulderHeight);
    }
    else if (type == ToolType::Sphere)
    {
        addLeaf(LeafKind::Radius, params.radius);
    }
    else
    {
        addLeaf(LeafKind::Radius, params.radius);
        addLeaf(LeafKind::CuttingLength, params.cuttingLength);
    }
    addLeaf(LeafKind::ShankLength, params.shankLength);
    addLeaf(LeafKind::ShankRadius, params.shankRadius);
    _toolTree->blockSignals(blocked);
}

QTreeWidgetItem* ToolManagerDialog::toolItemFromAny(QTreeWidgetItem* item) const
{
    if (!item) return nullptr;
    if (!item->parent()) return item;
    return item->parent();
}

ToolManagerDialog::ToolParams ToolManagerDialog::readToolParams(QTreeWidgetItem* toolItem) const
{
    ToolParams params;
    if (!toolItem) return params;
    for (int i = 0; i < toolItem->childCount(); ++i)
    {
        QTreeWidgetItem* leaf = toolItem->child(i);
        if (!leaf) continue;
        const auto kind = static_cast<LeafKind>(leaf->data(0, kRoleLeafKind).toInt());
        const double value = leaf->data(0, kRoleValue).toDouble();
        switch (kind)
        {
        case LeafKind::Radius: params.radius = value; break;
        case LeafKind::CuttingLength: params.cuttingLength = value; break;
        case LeafKind::ShankLength: params.shankLength = value; break;
        case LeafKind::ShankRadius: params.shankRadius = value; break;
        case LeafKind::TipWidth: params.tipWidth = value; break;
        case LeafKind::ShoulderWidth: params.shoulderWidth = value; break;
        case LeafKind::TaperHeight: params.taperHeight = value; break;
        case LeafKind::ShoulderHeight: params.shoulderHeight = value; break;
        }
    }
    return params;
}

void ToolManagerDialog::writeToolParams(QTreeWidgetItem* toolItem, const ToolParams& params)
{
    if (!toolItem || !_toolTree) return;
    const bool blocked = _toolTree->blockSignals(true);
    for (int i = 0; i < toolItem->childCount(); ++i)
    {
        QTreeWidgetItem* leaf = toolItem->child(i);
        if (!leaf) continue;
        const auto kind = static_cast<LeafKind>(leaf->data(0, kRoleLeafKind).toInt());
        switch (kind)
        {
        case LeafKind::Radius: setLeafValue(leaf, params.radius); break;
        case LeafKind::CuttingLength: setLeafValue(leaf, params.cuttingLength); break;
        case LeafKind::ShankLength: setLeafValue(leaf, params.shankLength); break;
        case LeafKind::ShankRadius: setLeafValue(leaf, params.shankRadius); break;
        case LeafKind::TipWidth: setLeafValue(leaf, params.tipWidth); break;
        case LeafKind::ShoulderWidth: setLeafValue(leaf, params.shoulderWidth); break;
        case LeafKind::TaperHeight: setLeafValue(leaf, params.taperHeight); break;
        case LeafKind::ShoulderHeight: setLeafValue(leaf, params.shoulderHeight); break;
        }
    }
    _toolTree->blockSignals(blocked);
}

void ToolManagerDialog::previewToolItem(QTreeWidgetItem* toolItem)
{
    if (!toolItem) return;
    const auto type = static_cast<ToolType>(toolItem->data(0, kRoleType).toInt());
    if (type == ToolType::None) return;
    const ToolParams params = readToolParams(toolItem);
    if (!_ready)
    {
        _pending = {true, type, params.radius, params.cuttingLength, params.shankLength,
                    params.shankRadius, params.tipWidth, params.shoulderWidth, params.taperHeight,
                    params.shoulderHeight, toolColor(toolItem)};
        return;
    }
    applyPreview(type, params.radius, params.cuttingLength, params.shankLength,
                 params.shankRadius, params.tipWidth, params.shoulderWidth, params.taperHeight,
                 params.shoulderHeight, toolColor(toolItem));
}

void ToolManagerDialog::onLibrarySelectionChanged()
{
    updateLibraryActions();
    if (!_toolTree) return;
    const auto selected = _toolTree->selectedItems();
    if (selected.isEmpty()) return;
    previewToolItem(toolItemFromAny(selected.front()));
}

void ToolManagerDialog::updateLibraryActions()
{
    const bool have = currentToolItem() && _toolTree && !_toolTree->selectedItems().isEmpty();
    if (_deleteButton) _deleteButton->setEnabled(have);
    if (_colorButton)
    {
        _colorButton->setEnabled(have);
        const QColor color = have ? toolColor(currentToolItem()) : kDefaultCutQColor;
        const int luma = (color.red() * 299 + color.green() * 587 + color.blue() * 114) / 1000;
        const QString fg = luma > 140 ? QStringLiteral("#111111") : QStringLiteral("#f4f4f4");
        _colorButton->setStyleSheet(
            QStringLiteral("QPushButton { background-color: %1; color: %2; }")
                .arg(color.name(), fg));
    }
}

void ToolManagerDialog::onNewTool()
{
    if (!_toolTree) return;
    auto* tool = addToolRow(nextToolId(), ToolType::FlatNose,
                            defaultLibraryParams(ToolType::FlatNose), kDefaultCutQColor);
    if (!tool) return;
    _toolTree->setCurrentItem(tool);
    previewToolItem(tool);
    updateLibraryActions();
    emitLibrarySnapshot(tool);
}

void ToolManagerDialog::onPickColor()
{
    QTreeWidgetItem* tool = currentToolItem();
    if (!tool) return;
    const QColor chosen = QColorDialog::getColor(toolColor(tool), this, "Tool color");
    if (!chosen.isValid()) return;
    tool->setData(0, kRoleColor, chosen);
    updateLibraryActions();
    previewToolItem(tool);
    emit toolColorChanged(tool->data(0, kRoleId).toInt(), tool->data(0, kRoleType).toInt(),
                         chosen);
}

void ToolManagerDialog::onToolTypeChanged(QTreeWidgetItem* toolItem)
{
    if (!toolItem || !_toolTree) return;
    auto* combo = qobject_cast<QComboBox*>(_toolTree->itemWidget(toolItem, 1));
    if (!combo) return;
    const auto type = static_cast<ToolType>(combo->currentData().toInt());
    if (type == ToolType::None) return;
    if (toolItem->data(0, kRoleType).toInt() == static_cast<int>(type)) return;
    toolItem->setData(0, kRoleType, static_cast<int>(type));
    rebuildLeaves(toolItem);
    previewToolItem(toolItem);
    emitLibrarySnapshot(toolItem);
}

void ToolManagerDialog::deleteSelectedTool()
{
    if (!_toolTree) return;
    if (QWidget* focus = QApplication::focusWidget())
    {
        if (qobject_cast<QLineEdit*>(focus) && _toolTree->isAncestorOf(focus))
            return;
    }

    QTreeWidgetItem* tool = toolItemFromAny(_toolTree->currentItem());
    if (!tool) return;
    const int index = _toolTree->indexOfTopLevelItem(tool);
    if (index < 0) return;
    const int toolId = tool->data(0, kRoleId).toInt();

    delete _toolTree->takeTopLevelItem(index);
    if (toolId > 0) emit toolLibraryRemoved(toolId);

    const int remaining = _toolTree->topLevelItemCount();
    if (remaining == 0)
    {
        updateLibraryActions();
        if (_ready)
            presentPreview(makePlaceholder(_options));
        else
            _pending = {};
        return;
    }

    _toolTree->setCurrentItem(_toolTree->topLevelItem(std::min(index, remaining - 1)));
}

void ToolManagerDialog::onLibraryItemDoubleClicked(QTreeWidgetItem* item, int)
{
    if (!item || !_toolTree) return;
    if (!item->parent())
    {
        item->setExpanded(!item->isExpanded());
        previewToolItem(item);
        return;
    }
    _toolTree->editItem(item, 0);
}

void ToolManagerDialog::onLibraryItemChanged(QTreeWidgetItem* item, int)
{
    if (!_toolTree || !item || !item->parent()) return;

    const auto kind = static_cast<LeafKind>(item->data(0, kRoleLeafKind).toInt());
    QString raw = item->text(0);
    const int colon = raw.indexOf(':');
    if (colon >= 0) raw = raw.mid(colon + 1);

    double value = 0.0;
    bool ok = false;
    if (kind == LeafKind::ShoulderHeight)
        ok = parseNonNegative(raw, &value);
    else
        ok = parsePositive(raw, &value);

    if (!ok)
    {
        value = item->data(0, kRoleValue).toDouble();
        if (kind == LeafKind::ShoulderHeight)
        {
            if (value < 0.0) value = 0.0;
        }
        else if (!(value > 0.0))
        {
            const Parameter& store = Parameter::instance();
            if (kind == LeafKind::TipWidth)
                value = store.toolWheelTipWidth() > 0.0 ? store.toolWheelTipWidth() : 0.01;
            else if (kind == LeafKind::ShoulderWidth)
                value = store.toolWheelShoulderWidth() > 0.0 ? store.toolWheelShoulderWidth()
                                                             : 0.06;
            else if (kind == LeafKind::TaperHeight)
                value = store.toolWheelTaperHeight() > 0.0 ? store.toolWheelTaperHeight() : 0.035;
            else if (kind == LeafKind::Radius || kind == LeafKind::ShankRadius)
                value = store.toolRadius() > 0.0 ? store.toolRadius() : 0.05;
            else
                value = store.toolLength() > 0.0 ? store.toolLength() : 0.14;
        }
    }

    const bool blocked = _toolTree->blockSignals(true);
    setLeafValue(item, value);
    _toolTree->blockSignals(blocked);

    QTreeWidgetItem* toolItem = item->parent();
    previewToolItem(toolItem);
    emitLibrarySnapshot(toolItem);
}

void ToolManagerDialog::emitLibrarySnapshot(QTreeWidgetItem* toolItem)
{
    if (!toolItem) return;
    const int toolId = toolItem->data(0, kRoleId).toInt();
    const auto type = static_cast<ToolType>(toolItem->data(0, kRoleType).toInt());
    if (toolId <= 0 || type == ToolType::None) return;
    const ToolParams params = readToolParams(toolItem);
    const double totalH = params.taperHeight + params.shoulderHeight;
    const double radius =
        (type == ToolType::GrindingWheel && totalH > 0.0) ? totalH : params.radius;
    const double cutting =
        (type == ToolType::GrindingWheel && totalH > 0.0) ? totalH : params.cuttingLength;
    emit toolLibraryEntryChanged(toolId, static_cast<int>(type), radius, cutting,
                                 params.shankLength, params.shankRadius, params.tipWidth,
                                 params.shoulderWidth, params.taperHeight, params.shoulderHeight);
}

void ToolManagerDialog::selectToolForId(int toolId)
{
    if (!_toolTree || toolId <= 0) return;
    for (int i = 0; i < _toolTree->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem* item = _toolTree->topLevelItem(i);
        if (!item || item->data(0, kRoleId).toInt() != toolId) continue;
        _toolTree->setCurrentItem(item);
        return;
    }
}

void ToolManagerDialog::selectToolForType(ToolType type)
{
    if (!_toolTree) return;
    const int wanted = static_cast<int>(type);
    for (int i = 0; i < _toolTree->topLevelItemCount(); ++i)
    {
        QTreeWidgetItem* item = _toolTree->topLevelItem(i);
        if (!item || item->data(0, kRoleType).toInt() != wanted) continue;
        _toolTree->setCurrentItem(item);
        return;
    }
}

void ToolManagerDialog::applyPreview(ToolType type, double radius, double cuttingLength,
                                     double shankLength, double shankRadius, double tipWidth,
                                     double shoulderWidth, double taperHeight,
                                     double shoulderHeight, const QColor& cutColor)
{
    if (type == ToolType::None)
    {
        presentPreview(makePlaceholder(_options));
        return;
    }

    if (!(radius > 0.0)) radius = 0.05;
    if (!(cuttingLength > 0.0)) cuttingLength = radius * 2.8;
    if (!(shankLength > 0.0)) shankLength = cuttingLength;
    if (!(shankRadius > 0.0)) shankRadius = radius;
    if (!(tipWidth > 0.0)) tipWidth = 0.01;
    if (!(shoulderWidth > 0.0)) shoulderWidth = 0.06;
    if (!(taperHeight > 0.0)) taperHeight = 0.035;
    if (shoulderHeight < 0.0) shoulderHeight = 0.0;
    if (type == ToolType::Sphere && shankRadius >= radius)
        shankRadius = radius * 0.6;

    GrindingWheelProfile wheel;
    wheel.tipWidth = static_cast<float>(tipWidth);
    wheel.shoulderWidth = static_cast<float>(shoulderWidth);
    wheel.taperHeight = static_cast<float>(taperHeight);
    wheel.shoulderHeight = static_cast<float>(shoulderHeight);
    if (type == ToolType::GrindingWheel && !wheel.valid()) wheel = GrindingWheelProfile{};

    const auto cutterRadius = static_cast<float>(radius);
    const auto cutterLength = static_cast<float>(cuttingLength);
    const TriangleMesh cutMesh =
        createToolMesh(type, cutterRadius, cutterLength, 48, 24, 12, 60.0f,
                       static_cast<float>(shankRadius), static_cast<float>(shankLength), wheel);
    auto cutNode = makeToolNode(cutMesh, _options, cutColorVec(cutColor), false);

    const auto group = vsg::Group::create();
    if (cutNode) group->addChild(cutNode);

    if (shankRadius > 0.0 && shankLength > 0.0)
    {
        TriangleMesh shankMesh;
        if (type == ToolType::GrindingWheel)
            shankMesh = createGrindingShankMesh(wheel.shoulderHeight, static_cast<float>(shankRadius),
                                               static_cast<float>(shankLength));
        else
        {
            const float z0 = (type == ToolType::Sphere)
                                 ? cutterRadius
                                 : toolCuttingTop(type, cutterRadius, cutterLength);
            shankMesh = createShankMesh(static_cast<float>(shankRadius), z0,
                                        static_cast<float>(shankLength));
        }
        if (auto shankNode = makeToolNode(shankMesh, _options, kMetalColor, true))
            group->addChild(shankNode);
    }

    if (group->children.empty())
        presentPreview(makePlaceholder(_options));
    else
        presentPreview(group);
}

void ToolManagerDialog::presentPreview(vsg::ref_ptr<vsg::Node> node)
{
    _scene->children.clear();
    if (node) _scene->addChild(node);
    frameScene();

    if (!_ready || !_viewer) return;
    if (_viewer->compileManager && node)
    {
        auto compiled = _viewer->compileManager->compile(node);
        if (compiled) vsg::updateViewer(*_viewer, compiled);
    }
    _viewer->request();
}

void ToolManagerDialog::frameScene()
{
    if (!_scene || !_lookAt) return;

    vsg::ComputeBounds computeBounds;
    _scene->accept(computeBounds);
    const vsg::dvec3 centre =
        (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
    const double radius =
        vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
    const double viewRadius = (radius > 1.0e-6) ? radius : 1.0;

    _lookAt->center = centre;
    _lookAt->eye = centre + vsg::dvec3(0.0, -viewRadius * 3.5, 0.0);
    _lookAt->up = vsg::dvec3(0.0, 0.0, 1.0);
    if (_perspective)
    {
        _perspective->nearDistance = 0.001 * viewRadius;
        _perspective->farDistance = viewRadius * 4.5;
    }
}

void ToolManagerDialog::initializeScene()
{
    if (_ready) return;

    try
    {
        _window->initializeWindow();
        if (!_window->windowAdapter)
            throw vsg::Exception{"Failed to create the Tool manager Vulkan surface.",
                                 VK_ERROR_INITIALIZATION_FAILED};

        if (!_traits->device)
            _traits->device = _window->windowAdapter->getOrCreateDevice();

        if (_pending.valid)
            applyPreview(_pending.type, _pending.radius, _pending.cuttingLength,
                         _pending.shankLength, _pending.shankRadius, _pending.tipWidth,
                         _pending.shoulderWidth, _pending.taperHeight, _pending.shoulderHeight,
                         _pending.cutColor);
        else
            _scene->addChild(makePlaceholder(_options));
        _pending = {};

        vsg::ComputeBounds computeBounds;
        _scene->accept(computeBounds);
        const vsg::dvec3 centre =
            (computeBounds.bounds.min + computeBounds.bounds.max) * 0.5;
        const double radius =
            vsg::length(computeBounds.bounds.max - computeBounds.bounds.min) * 0.6;
        const double viewRadius = (radius > 1.0e-6) ? radius : 1.0;

        const uint32_t width = _window->traits->width;
        const uint32_t height = _window->traits->height;
        const double aspect = (height > 0)
                                  ? static_cast<double>(width) / static_cast<double>(height)
                                  : 1.0;

        _lookAt = vsg::LookAt::create(centre + vsg::dvec3(0.0, -viewRadius * 3.5, 0.0),
                                      centre, vsg::dvec3(0.0, 0.0, 1.0));
        _perspective = vsg::Perspective::create(30.0, aspect, 0.001 * viewRadius,
                                                viewRadius * 4.5);
        auto camera = vsg::Camera::create(
            _perspective, _lookAt, vsg::ViewportState::create(VkExtent2D{width, height}));

        auto trackball = vsg::Trackball::create(camera);
        trackball->addWindow(*_window);
        _viewer->addEventHandler(trackball);

        auto renderGraph = vsg::createRenderGraphForView(*_window, camera, _scene);
        auto commandGraph = vsg::CommandGraph::create(*_window);
        commandGraph->addChild(renderGraph);
        _viewer->addRecordAndSubmitTaskAndPresentation({commandGraph});
        _viewer->compile();

        if (_timerInterval >= 0) _viewer->setInterval(_timerInterval);
        _viewer->continuousUpdate = true;
        _ready = true;
    }
    catch (const vsg::Exception& e)
    {
        QMessageBox::warning(this, "Tool manager",
                             QString("Could not start the 3D view: %1")
                                 .arg(QString::fromStdString(e.message)));
    }
}

} // namespace app
