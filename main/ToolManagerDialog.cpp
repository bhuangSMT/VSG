#include "ToolManagerDialog.h"
#include "UcamVsgGuard.h"

#include <algorithm>

#include <QtGui/QCloseEvent>
#include <QtGui/QFont>
#include <QtGui/QShowEvent>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMessageBox>
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

const vsg::vec4 kCutColor{0.95f, 0.35f, 0.10f, 1.0f};
const vsg::vec4 kMetalColor{0.78f, 0.80f, 0.84f, 1.0f};

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

    _toolTree = new QTreeWidget();
    _toolTree->setHeaderLabel("Tools");
    _toolTree->setRootIsDecorated(true);
    _toolTree->setUniformRowHeights(true);
    _toolTree->setSelectionMode(QAbstractItemView::SingleSelection);
    _toolTree->setEditTriggers(QAbstractItemView::DoubleClicked |
                               QAbstractItemView::EditKeyPressed |
                               QAbstractItemView::AnyKeyPressed);
    _toolTree->setColumnCount(1);
    _toolTree->header()->setStretchLastSection(true);
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

void ToolManagerDialog::previewTool(ToolType type, double radius, double cuttingLength,
                                    double shankLength, double shankRadius, double tipWidth,
                                    double shoulderWidth, double taperHeight,
                                    double shoulderHeight)
{
    selectToolForType(type);
    QTreeWidgetItem* toolItem = nullptr;
    if (_toolTree)
    {
        for (int i = 0; i < _toolTree->topLevelItemCount(); ++i)
        {
            QTreeWidgetItem* item = _toolTree->topLevelItem(i);
            if (item && item->data(0, kRoleType).toInt() == static_cast<int>(type))
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
    if (!_ready)
    {
        _pending = {true, type, radius, cuttingLength, shankLength, shankRadius,
                    tipWidth, shoulderWidth, taperHeight, shoulderHeight};
        return;
    }
    applyPreview(type, radius, cuttingLength, shankLength, shankRadius, tipWidth, shoulderWidth,
                 taperHeight, shoulderHeight);
}

void ToolManagerDialog::fillToolLibrary()
{
    if (!_toolTree) return;

    const Parameter& store = Parameter::instance();
    const double radius = store.toolRadius() > 0.0 ? store.toolRadius() : 0.05;
    const double length = store.toolLength() > 0.0 ? store.toolLength() : radius * 2.8;
    const double tipWidth = store.toolWheelTipWidth() > 0.0 ? store.toolWheelTipWidth() : 0.01;
    const double shoulderWidth =
        store.toolWheelShoulderWidth() > 0.0 ? store.toolWheelShoulderWidth() : 0.06;
    const double taperHeight =
        store.toolWheelTaperHeight() > 0.0 ? store.toolWheelTaperHeight() : 0.035;
    const double shoulderHeight =
        store.toolWheelShoulderHeight() >= 0.0 ? store.toolWheelShoulderHeight() : 0.015;

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
    {
        auto* tool = new QTreeWidgetItem();
        tool->setFlags((tool->flags() | Qt::ItemIsEnabled | Qt::ItemIsSelectable) &
                       ~Qt::ItemIsEditable);
        tool->setData(0, kRoleType, static_cast<int>(seed.type));
        tool->setText(0, QString("%1 %2").arg(seed.id).arg(
                             QString::fromLatin1(toolTypeLabel(seed.type))));

        const double shankRadius =
            (seed.type == ToolType::Sphere) ? radius * 0.6 : radius * 1.2;
        ToolParams params{radius, length, length, shankRadius, tipWidth, shoulderWidth,
                          taperHeight, shoulderHeight};
        if (seed.type == ToolType::GrindingWheel)
        {
            params.cuttingLength = params.taperHeight + params.shoulderHeight;
            params.radius = params.cuttingLength;
            tool->addChild(makeLeaf(LeafKind::TipWidth, params.tipWidth));
            tool->addChild(makeLeaf(LeafKind::ShoulderWidth, params.shoulderWidth));
            tool->addChild(makeLeaf(LeafKind::TaperHeight, params.taperHeight));
            tool->addChild(makeLeaf(LeafKind::ShoulderHeight, params.shoulderHeight));
        }
        else if (seed.type == ToolType::Sphere)
        {
            tool->addChild(makeLeaf(LeafKind::Radius, params.radius));
        }
        else
        {
            tool->addChild(makeLeaf(LeafKind::Radius, params.radius));
            tool->addChild(makeLeaf(LeafKind::CuttingLength, params.cuttingLength));
        }
        tool->addChild(makeLeaf(LeafKind::ShankLength, params.shankLength));
        tool->addChild(makeLeaf(LeafKind::ShankRadius, params.shankRadius));
        tool->setExpanded(false);
        _toolTree->addTopLevelItem(tool);
    }
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
                    params.shoulderHeight};
        return;
    }
    applyPreview(type, params.radius, params.cuttingLength, params.shankLength,
                 params.shankRadius, params.tipWidth, params.shoulderWidth, params.taperHeight,
                 params.shoulderHeight);
}

void ToolManagerDialog::onLibrarySelectionChanged()
{
    if (!_toolTree) return;
    const auto selected = _toolTree->selectedItems();
    if (selected.isEmpty()) return;
    previewToolItem(toolItemFromAny(selected.front()));
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

    const auto type = static_cast<ToolType>(toolItem->data(0, kRoleType).toInt());
    const ToolParams params = readToolParams(toolItem);
    const double totalH = params.taperHeight + params.shoulderHeight;
    const double radius =
        (type == ToolType::GrindingWheel && totalH > 0.0) ? totalH : params.radius;
    const double cutting =
        (type == ToolType::GrindingWheel && totalH > 0.0) ? totalH : params.cuttingLength;
    emit toolLibraryEntryChanged(static_cast<int>(type), radius, cutting, params.shankLength,
                                 params.shankRadius, params.tipWidth, params.shoulderWidth,
                                 params.taperHeight, params.shoulderHeight);
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
                                     double shoulderHeight)
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
    auto cutNode = makeToolNode(cutMesh, _options, kCutColor, false);

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
                         _pending.shoulderWidth, _pending.taperHeight, _pending.shoulderHeight);
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
