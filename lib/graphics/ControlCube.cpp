#include "ControlCube.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <QFont>
#include <QImage>
#include <QPainter>
#include <QWindow>

namespace app
{
namespace
{

constexpr float cubeHalf = 1.0f;
constexpr double cubeEyeDistance = 3.0;
constexpr double cubeOrthoHalf = 2.15;
constexpr double snapDuration = 0.4;
constexpr int stillClickSlopSq = 36;
constexpr double handleSphereRadius = 0.16;
constexpr double handleTubeRadius = 0.10;
constexpr double handleTubeEndGap = 0.18;
constexpr double pi = 3.14159265358979323846;

struct CornerHandle
{
    vsg::dvec3 center;
    vsg::dvec3 snap;
};

struct EdgeHandle
{
    vsg::dvec3 a;
    vsg::dvec3 b;
    vsg::dvec3 snap;
};

const std::array<CornerHandle, 8>& cornerHandles()
{
    static const std::array<CornerHandle, 8> handles = {{
        {{-1.0, -1.0, -1.0}, vsg::normalize(vsg::dvec3(-1.0, -1.0, -1.0))},
        {{1.0, -1.0, -1.0}, vsg::normalize(vsg::dvec3(1.0, -1.0, -1.0))},
        {{-1.0, 1.0, -1.0}, vsg::normalize(vsg::dvec3(-1.0, 1.0, -1.0))},
        {{1.0, 1.0, -1.0}, vsg::normalize(vsg::dvec3(1.0, 1.0, -1.0))},
        {{-1.0, -1.0, 1.0}, vsg::normalize(vsg::dvec3(-1.0, -1.0, 1.0))},
        {{1.0, -1.0, 1.0}, vsg::normalize(vsg::dvec3(1.0, -1.0, 1.0))},
        {{-1.0, 1.0, 1.0}, vsg::normalize(vsg::dvec3(-1.0, 1.0, 1.0))},
        {{1.0, 1.0, 1.0}, vsg::normalize(vsg::dvec3(1.0, 1.0, 1.0))},
    }};
    return handles;
}

const std::array<EdgeHandle, 12>& edgeHandles()
{
    const double g = handleTubeEndGap;
    auto edge = [](double ax, double ay, double az, double bx, double by, double bz,
                   double sx, double sy, double sz) {
        return EdgeHandle{{ax, ay, az}, {bx, by, bz}, vsg::normalize(vsg::dvec3(sx, sy, sz))};
    };

    static const std::array<EdgeHandle, 12> handles = {{
        edge(-1.0 + g, -1.0, -1.0, 1.0 - g, -1.0, -1.0, 0.0, -1.0, -1.0),
        edge(-1.0 + g, 1.0, -1.0, 1.0 - g, 1.0, -1.0, 0.0, 1.0, -1.0),
        edge(-1.0 + g, -1.0, 1.0, 1.0 - g, -1.0, 1.0, 0.0, -1.0, 1.0),
        edge(-1.0 + g, 1.0, 1.0, 1.0 - g, 1.0, 1.0, 0.0, 1.0, 1.0),
        edge(-1.0, -1.0 + g, -1.0, -1.0, 1.0 - g, -1.0, -1.0, 0.0, -1.0),
        edge(1.0, -1.0 + g, -1.0, 1.0, 1.0 - g, -1.0, 1.0, 0.0, -1.0),
        edge(-1.0, -1.0 + g, 1.0, -1.0, 1.0 - g, 1.0, -1.0, 0.0, 1.0),
        edge(1.0, -1.0 + g, 1.0, 1.0, 1.0 - g, 1.0, 1.0, 0.0, 1.0),
        edge(-1.0, -1.0, -1.0 + g, -1.0, -1.0, 1.0 - g, -1.0, -1.0, 0.0),
        edge(1.0, -1.0, -1.0 + g, 1.0, -1.0, 1.0 - g, 1.0, -1.0, 0.0),
        edge(-1.0, 1.0, -1.0 + g, -1.0, 1.0, 1.0 - g, -1.0, 1.0, 0.0),
        edge(1.0, 1.0, -1.0 + g, 1.0, 1.0, 1.0 - g, 1.0, 1.0, 0.0),
    }};
    return handles;
}

struct MeshBuilder
{
    std::vector<vsg::vec3> positions;
    std::vector<vsg::vec3> normals;
    std::vector<vsg::vec4> colors;
    std::vector<uint32_t> indices;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;

    uint32_t addVertex(const vsg::vec3& p, const vsg::vec3& n, const vsg::vec4& c)
    {
        positions.push_back(p);
        normals.push_back(n);
        colors.push_back(c);
        const uint32_t id = vertexCount;
        ++vertexCount;
        return id;
    }

    void addTri(uint32_t a, uint32_t b, uint32_t c)
    {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
        indexCount += 3;
    }
};

vsg::ref_ptr<vsg::Node> buildPhong(const MeshBuilder& mesh,
                                   vsg::ref_ptr<vsg::Options> options,
                                   bool transparent = false,
                                   vsg::ref_ptr<vsg::Data> texture = {},
                                   vsg::ref_ptr<vsg::vec2Array> texcoords = {})
{
    auto positions = vsg::vec3Array::create(mesh.vertexCount);
    auto normals = vsg::vec3Array::create(mesh.vertexCount);
    auto colors = vsg::vec4Array::create(mesh.vertexCount);
    auto indices = vsg::uintArray::create(mesh.indexCount);
    for (uint32_t i = 0; i < mesh.vertexCount; ++i)
    {
        (*positions)[i] = mesh.positions[i];
        (*normals)[i] = mesh.normals[i];
        (*colors)[i] = mesh.colors[i];
    }
    for (uint32_t i = 0; i < mesh.indexCount; ++i)
        (*indices)[i] = mesh.indices[i];

    if (!texcoords)
        texcoords = vsg::vec2Array::create(mesh.vertexCount, vsg::vec2(0.0f, 0.0f));

    auto shaderSet = vsg::createPhongShaderSet(options);
    auto gpc = vsg::GraphicsPipelineConfigurator::create(shaderSet);

    if (const auto& materialBinding = shaderSet->getDescriptorBinding("material"))
    {
        vsg::ref_ptr<vsg::Data> mat = materialBinding.data;
        if (texture)
        {
            auto material = vsg::PhongMaterialValue::create();
            auto& phong = material->value();
            phong.ambient = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            phong.diffuse = vsg::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            phong.emissive = vsg::vec4(0.35f, 0.35f, 0.35f, 1.0f);
            phong.alphaMask = 1.0f;
            phong.alphaMaskCutoff = 0.1f;
            mat = material;
        }
        else if (!mat)
        {
            mat = vsg::PhongMaterialValue::create();
        }
        gpc->assignDescriptor("material", mat);
    }

    if (texture)
    {
        auto sampler = vsg::Sampler::create();
        sampler->addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler->addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler->maxLod = 0.0f;
        gpc->assignTexture("diffuseMap", texture, sampler);
    }

    gpc->enableArray("vsg_Vertex", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_Normal", VK_VERTEX_INPUT_RATE_VERTEX, 12);
    gpc->enableArray("vsg_TexCoord0", VK_VERTEX_INPUT_RATE_VERTEX, 8);
    gpc->enableArray("vsg_Color", VK_VERTEX_INPUT_RATE_VERTEX, 16);

    if (transparent)
    {
        struct SetBlendState : public vsg::Visitor
        {
            void apply(vsg::Object& object) override { object.traverse(*this); }
            void apply(vsg::RasterizationState& rs) override
            {
                rs.cullMode = VK_CULL_MODE_NONE;
            }
            void apply(vsg::ColorBlendState& cbs) override
            {
                cbs.configureAttachments(true);
            }
        } setBlendState;
        gpc->accept(setBlendState);
    }

    gpc->init();

    auto vid = vsg::VertexIndexDraw::create();
    vid->assignArrays(vsg::DataList{positions, normals, texcoords, colors});
    vid->assignIndices(indices);
    vid->indexCount = mesh.indexCount;
    vid->instanceCount = 1;

    vsg::StateCommands stateCommands;
    if (!gpc->copyTo(stateCommands))
        throw std::runtime_error("Failed to create control-cube pipeline.");

    auto stateGroup = vsg::StateGroup::create();
    stateGroup->stateCommands.swap(stateCommands);
    stateGroup->prototypeArrayState = gpc->getSuitableArrayState();
    stateGroup->addChild(vid);
    return stateGroup;
}

void addFace(MeshBuilder& mesh,
             const vsg::vec3& v0, const vsg::vec3& v1, const vsg::vec3& v2, const vsg::vec3& v3,
             const vsg::vec3& normal, const vsg::vec4& color)
{
    const uint32_t a = mesh.addVertex(v0, normal, color);
    const uint32_t b = mesh.addVertex(v1, normal, color);
    const uint32_t c = mesh.addVertex(v2, normal, color);
    const uint32_t d = mesh.addVertex(v3, normal, color);
    mesh.addTri(a, b, c);
    mesh.addTri(a, c, d);
}

void addUvSphere(MeshBuilder& mesh, const vsg::vec3& center, float radius, const vsg::vec4& color,
                 int slices, int stacks)
{
    const uint32_t base = mesh.vertexCount;
    for (int s = 0; s <= stacks; ++s)
    {
        const float v = static_cast<float>(s) / static_cast<float>(stacks);
        const float theta = v * static_cast<float>(pi);
        const float sinT = std::sin(theta);
        const float cosT = std::cos(theta);
        for (int i = 0; i <= slices; ++i)
        {
            const float u = static_cast<float>(i) / static_cast<float>(slices);
            const float phi = u * 2.0f * static_cast<float>(pi);
            const vsg::vec3 n(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
            mesh.addVertex(center + n * radius, n, color);
        }
    }

    const int stride = slices + 1;
    for (int s = 0; s < stacks; ++s)
    {
        for (int i = 0; i < slices; ++i)
        {
            const uint32_t i0 = base + static_cast<uint32_t>(s * stride + i);
            const uint32_t i1 = base + static_cast<uint32_t>(s * stride + i + 1);
            const uint32_t i2 = base + static_cast<uint32_t>((s + 1) * stride + i);
            const uint32_t i3 = base + static_cast<uint32_t>((s + 1) * stride + i + 1);
            mesh.addTri(i0, i2, i3);
            mesh.addTri(i0, i3, i1);
        }
    }
}

vsg::vec3 unitPerpendicular(const vsg::vec3& axis)
{
    const vsg::vec3 hint = (std::abs(axis.z) < 0.9f) ? vsg::vec3(0.0f, 0.0f, 1.0f)
                                                     : vsg::vec3(1.0f, 0.0f, 0.0f);
    return vsg::normalize(vsg::cross(axis, hint));
}

void addTube(MeshBuilder& mesh, const vsg::vec3& a, const vsg::vec3& b, float radius,
             const vsg::vec4& color, int slices)
{
    vsg::vec3 axis = b - a;
    const float len = vsg::length(axis);
    if (len <= 0.0f) return;
    axis /= len;
    const vsg::vec3 side = unitPerpendicular(axis);
    const vsg::vec3 up = vsg::normalize(vsg::cross(axis, side));

    const uint32_t base = mesh.vertexCount;
    for (int i = 0; i <= slices; ++i)
    {
        const float ang = static_cast<float>(i) / static_cast<float>(slices) * 2.0f * static_cast<float>(pi);
        const vsg::vec3 n = side * std::cos(ang) + up * std::sin(ang);
        mesh.addVertex(a + n * radius, n, color);
        mesh.addVertex(b + n * radius, n, color);
    }

    for (int i = 0; i < slices; ++i)
    {
        const uint32_t i0 = base + static_cast<uint32_t>(i * 2);
        const uint32_t i1 = base + static_cast<uint32_t>(i * 2 + 1);
        const uint32_t i2 = base + static_cast<uint32_t>((i + 1) * 2);
        const uint32_t i3 = base + static_cast<uint32_t>((i + 1) * 2 + 1);
        mesh.addTri(i0, i2, i3);
        mesh.addTri(i0, i3, i1);
    }
}

vsg::ref_ptr<vsg::ubvec4Array2D> rasterizeLabel(const char* text)
{
    constexpr int dim = 128;
    QImage image(dim, dim, QImage::Format_RGBA8888);
    image.fill(QColor(0, 0, 0, 0));

    QPainter painter(&image);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);
    QFont font;
    font.setBold(true);
    font.setPixelSize((text[0] != '\0' && text[1] != '\0') ? 42 : 72);
    painter.setFont(font);
    painter.setPen(QColor(255, 255, 255, 255));
    painter.drawText(image.rect(), Qt::AlignCenter, QString::fromLatin1(text));
    painter.end();

    auto data = vsg::ubvec4Array2D::create(static_cast<uint32_t>(dim), static_cast<uint32_t>(dim));
    data->properties.format = VK_FORMAT_R8G8B8A8_UNORM;
    data->properties.origin = vsg::TOP_LEFT;
    for (int y = 0; y < dim; ++y)
    {
        for (int x = 0; x < dim; ++x)
        {
            const QRgb px = image.pixel(x, y);
            (*data)(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) =
                vsg::ubvec4(static_cast<uint8_t>(qRed(px)),
                            static_cast<uint8_t>(qGreen(px)),
                            static_cast<uint8_t>(qBlue(px)),
                            static_cast<uint8_t>(qAlpha(px)));
        }
    }
    return data;
}

vsg::ref_ptr<vsg::Node> buildLetterQuad(const vsg::vec3& normal, const vsg::vec3& letterUp,
                                        const char* text, float halfWidth, float halfHeight,
                                        vsg::ref_ptr<vsg::Options> options)
{
    const vsg::vec3 n = vsg::normalize(normal);
    const vsg::vec3 up = vsg::normalize(letterUp);
    const vsg::vec3 right = vsg::normalize(vsg::cross(-n, up));
    const vsg::vec3 center = n * (cubeHalf + 0.03f);
    const vsg::vec3 v0 = center - right * halfWidth - up * halfHeight;
    const vsg::vec3 v1 = center + right * halfWidth - up * halfHeight;
    const vsg::vec3 v2 = center + right * halfWidth + up * halfHeight;
    const vsg::vec3 v3 = center - right * halfWidth + up * halfHeight;
    const vsg::vec4 white(1.0f, 1.0f, 1.0f, 1.0f);

    MeshBuilder mesh;
    const uint32_t a = mesh.addVertex(v0, n, white);
    const uint32_t b = mesh.addVertex(v1, n, white);
    const uint32_t c = mesh.addVertex(v2, n, white);
    const uint32_t d = mesh.addVertex(v3, n, white);
    mesh.addTri(a, b, c);
    mesh.addTri(a, c, d);

    auto texcoords = vsg::vec2Array::create(uint32_t{4});
    (*texcoords)[0] = vsg::vec2(0.0f, 1.0f);
    (*texcoords)[1] = vsg::vec2(1.0f, 1.0f);
    (*texcoords)[2] = vsg::vec2(1.0f, 0.0f);
    (*texcoords)[3] = vsg::vec2(0.0f, 0.0f);

    return buildPhong(mesh, options, true, rasterizeLabel(text), texcoords);
}

vsg::ref_ptr<vsg::Node> buildCubeMesh(vsg::ref_ptr<vsg::Options> options)
{
    MeshBuilder faces;
    const float h = cubeHalf;
    const vsg::vec4 plusX(0.85f, 0.18f, 0.18f, 1.0f);
    const vsg::vec4 minusX(0.45f, 0.10f, 0.10f, 1.0f);
    const vsg::vec4 plusY(0.18f, 0.75f, 0.22f, 1.0f);
    const vsg::vec4 minusY(0.10f, 0.40f, 0.12f, 1.0f);
    const vsg::vec4 plusZ(0.20f, 0.40f, 0.90f, 1.0f);
    const vsg::vec4 minusZ(0.10f, 0.20f, 0.50f, 1.0f);

    addFace(faces, {h, -h, -h}, {h, h, -h}, {h, h, h}, {h, -h, h}, {1.0f, 0.0f, 0.0f}, plusX);
    addFace(faces, {-h, -h, h}, {-h, h, h}, {-h, h, -h}, {-h, -h, -h}, {-1.0f, 0.0f, 0.0f}, minusX);
    addFace(faces, {-h, h, -h}, {-h, h, h}, {h, h, h}, {h, h, -h}, {0.0f, 1.0f, 0.0f}, plusY);
    addFace(faces, {-h, -h, h}, {-h, -h, -h}, {h, -h, -h}, {h, -h, h}, {0.0f, -1.0f, 0.0f}, minusY);
    addFace(faces, {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}, {0.0f, 0.0f, 1.0f}, plusZ);
    addFace(faces, {h, -h, -h}, {-h, -h, -h}, {-h, h, -h}, {h, h, -h}, {0.0f, 0.0f, -1.0f}, minusZ);

    MeshBuilder handles;
    const vsg::vec4 chrome(0.22f, 0.22f, 0.24f, 1.0f);
    for (const auto& corner : cornerHandles())
    {
        addUvSphere(handles,
                    vsg::vec3(static_cast<float>(corner.center.x),
                              static_cast<float>(corner.center.y),
                              static_cast<float>(corner.center.z)),
                    static_cast<float>(handleSphereRadius), chrome, 12, 8);
    }
    for (const auto& edge : edgeHandles())
    {
        addTube(handles,
                vsg::vec3(static_cast<float>(edge.a.x), static_cast<float>(edge.a.y),
                          static_cast<float>(edge.a.z)),
                vsg::vec3(static_cast<float>(edge.b.x), static_cast<float>(edge.b.y),
                          static_cast<float>(edge.b.z)),
                static_cast<float>(handleTubeRadius), chrome, 12);
    }

    auto group = vsg::Group::create();
    group->addChild(buildPhong(faces, options));
    group->addChild(buildPhong(handles, options));
    group->addChild(buildLetterQuad({1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "R", 0.42f, 0.42f, options));
    group->addChild(buildLetterQuad({-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "L", 0.42f, 0.42f, options));
    group->addChild(buildLetterQuad({0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "B", 0.42f, 0.42f, options));
    group->addChild(buildLetterQuad({0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, "F", 0.42f, 0.42f, options));
    group->addChild(buildLetterQuad({0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, "T", 0.42f, 0.42f, options));
    group->addChild(buildLetterQuad({0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}, "BOT", 0.62f, 0.36f, options));
    return group;
}

int32_t scaledPixels(int logical, double devicePixelRatio)
{
    const double scale = (devicePixelRatio > 0.0) ? devicePixelRatio : 1.0;
    const int value = static_cast<int>(std::lround(static_cast<double>(logical) * scale));
    return (value > 1) ? value : 1;
}

bool intersectSphere(const vsg::dvec3& origin, const vsg::dvec3& dir, const vsg::dvec3& center,
                     double radius, double tMax, double& tHit)
{
    const vsg::dvec3 oc = origin - center;
    const double b = vsg::dot(oc, dir);
    const double c = vsg::dot(oc, oc) - radius * radius;
    const double disc = b * b - c;
    if (disc < 0.0) return false;
    const double t = -b - std::sqrt(disc);
    if (t < 0.0 || t > tMax) return false;
    tHit = t;
    return true;
}

bool intersectTube(const vsg::dvec3& origin, const vsg::dvec3& dir, const EdgeHandle& edge,
                   double radius, double tMax, double& tHit)
{
    vsg::dvec3 axis = edge.b - edge.a;
    const double axisLen = vsg::length(axis);
    if (axisLen <= 0.0) return false;
    axis /= axisLen;

    const vsg::dvec3 m = origin - edge.a;
    const vsg::dvec3 dPerp = dir - axis * vsg::dot(dir, axis);
    const vsg::dvec3 mPerp = m - axis * vsg::dot(m, axis);
    const double a = vsg::dot(dPerp, dPerp);
    if (a < 1.0e-12)
    {
        if (vsg::dot(mPerp, mPerp) > radius * radius) return false;
        return false;
    }

    const double b = vsg::dot(dPerp, mPerp);
    const double c = vsg::dot(mPerp, mPerp) - radius * radius;
    const double disc = b * b - a * c;
    if (disc < 0.0) return false;
    const double t = (-b - std::sqrt(disc)) / a;
    if (t < 0.0 || t > tMax) return false;

    const vsg::dvec3 hit = origin + dir * t;
    const double along = vsg::dot(hit - edge.a, axis);
    if (along < 0.0 || along > axisLen) return false;
    tHit = t;
    return true;
}

bool intersectFace(const vsg::dvec3& origin, const vsg::dvec3& dir, double tMax,
                   vsg::dvec3& snap, double& tHit)
{
    double tMin = 0.0;
    double tFar = tMax;
    auto slab = [&](double orig, double d) {
        if (std::abs(d) < 1.0e-12)
            return orig >= -1.0 && orig <= 1.0;
        const double invD = 1.0 / d;
        double t0 = (-1.0 - orig) * invD;
        double t1 = (1.0 - orig) * invD;
        if (t0 > t1) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tFar = std::min(tFar, t1);
        return tMin <= tFar;
    };

    if (!slab(origin.x, dir.x) || !slab(origin.y, dir.y) || !slab(origin.z, dir.z))
        return false;

    const vsg::dvec3 hit = origin + dir * tMin;
    const double ax = std::abs(hit.x);
    const double ay = std::abs(hit.y);
    const double az = std::abs(hit.z);
    if (ax >= ay && ax >= az)
        snap = vsg::dvec3(hit.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0);
    else if (ay >= az)
        snap = vsg::dvec3(0.0, hit.y >= 0.0 ? 1.0 : -1.0, 0.0);
    else
        snap = vsg::dvec3(0.0, 0.0, hit.z >= 0.0 ? 1.0 : -1.0);
    tHit = tMin;
    return true;
}

} // namespace

ControlCube::ControlCube(vsg::ref_ptr<vsg::Camera> mainCamera,
                         vsg::ref_ptr<vsg::Trackball> trackball,
                         vsg::ref_ptr<vsg::Options> options,
                         vsg::ref_ptr<vsg::Window> window,
                         QWindow* qtWindow) :
    _mainCamera(std::move(mainCamera)),
    _trackball(std::move(trackball)),
    _window(std::move(window)),
    _qtWindow(qtWindow)
{
    _cubeLookAt = vsg::LookAt::create(vsg::dvec3(0.0, -cubeEyeDistance, 0.0),
                                      vsg::dvec3(0.0, 0.0, 0.0),
                                      vsg::dvec3(0.0, 0.0, 1.0));
    _cubeOrtho = vsg::Orthographic::create(-cubeOrthoHalf, cubeOrthoHalf,
                                           -cubeOrthoHalf, cubeOrthoHalf,
                                           0.1, 10.0);
    _cubeViewport = vsg::ViewportState::create(insetMargin, insetMargin,
                                               static_cast<uint32_t>(insetSize),
                                               static_cast<uint32_t>(insetSize));
    _cubeCamera = vsg::Camera::create(_cubeOrtho, _cubeLookAt, _cubeViewport);

    _view = vsg::View::create(_cubeCamera);
    _view->addChild(vsg::createHeadlight());
    _view->addChild(buildCubeMesh(options));

    VkClearAttachment attachment{};
    attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    attachment.clearValue.depthStencil = {0.0f, 0};

    VkClearRect rect{};
    rect.rect.offset = {insetMargin, insetMargin};
    rect.rect.extent = {static_cast<uint32_t>(insetSize), static_cast<uint32_t>(insetSize)};
    rect.baseArrayLayer = 0;
    rect.layerCount = 1;

    _depthClear = vsg::ClearAttachments::create(
        vsg::ClearAttachments::Attachments{attachment},
        vsg::ClearAttachments::Rects{rect});

    updateInset();
    syncFromMain();
}

bool ControlCube::contains(int32_t x, int32_t y) const
{
    return x >= _insetX && x < _insetX + _insetExtent &&
           y >= _insetY && y < _insetY + _insetExtent;
}

void ControlCube::apply(vsg::FrameEvent&)
{
    updateInset();
    syncFromMain();
}

void ControlCube::apply(vsg::ButtonPressEvent& press)
{
    _lastX = press.x;
    _lastY = press.y;
    if (!contains(press.x, press.y)) return;

    claimPointer(press);
    _pressInCube = true;
    _pressX = press.x;
    _pressY = press.y;
    _leftPressed = (press.button == 1);
    _dragging = false;
    _prevTbc = insetTbc(press.x, press.y);
}

void ControlCube::apply(vsg::ButtonReleaseEvent& release)
{
    _lastX = release.x;
    _lastY = release.y;
    const bool pressInCube = _pressInCube;
    const bool leftPressed = _leftPressed;
    const bool wasDragging = _dragging;
    _pressInCube = false;
    _leftPressed = false;
    _dragging = false;

    if (!pressInCube && !contains(release.x, release.y)) return;
    claimPointer(release);

    if (!leftPressed || release.button != 1 || wasDragging) return;
    const int32_t dx = release.x - _pressX;
    const int32_t dy = release.y - _pressY;
    if (dx * dx + dy * dy > stillClickSlopSq) return;
    if (!contains(release.x, release.y)) return;

    if (const auto direction = pickSnapDirection(release.x, release.y))
        snapToDirection(*direction);
}

void ControlCube::apply(vsg::MoveEvent& move)
{
    _lastX = move.x;
    _lastY = move.y;
    if (_pressInCube || contains(move.x, move.y))
        claimPointer(move);

    if (!_leftPressed || !_pressInCube) return;

    const int32_t dx = move.x - _pressX;
    const int32_t dy = move.y - _pressY;
    if (!_dragging && dx * dx + dy * dy > stillClickSlopSq)
        _dragging = true;
    if (_dragging)
        orbitDrag(move.x, move.y);
}

void ControlCube::apply(vsg::ScrollWheelEvent& scroll)
{
    if (contains(_lastX, _lastY))
        claimPointer(scroll);
}

void ControlCube::claimPointer(vsg::UIEvent& event)
{
    event.handled = true;
}

void ControlCube::syncFromMain()
{
    if (!_mainCamera || !_cubeLookAt) return;
    auto* mainLookAt = dynamic_cast<vsg::LookAt*>(_mainCamera->viewMatrix.get());
    if (!mainLookAt) return;

    vsg::dvec3 look = mainLookAt->center - mainLookAt->eye;
    const double lookLen = vsg::length(look);
    if (lookLen <= 1.0e-12) return;
    look /= lookLen;

    _cubeLookAt->center = vsg::dvec3(0.0, 0.0, 0.0);
    _cubeLookAt->eye = -look * cubeEyeDistance;

    vsg::dvec3 up = mainLookAt->up;
    vsg::dvec3 side = vsg::cross(look, up);
    const double sideLen = vsg::length(side);
    if (sideLen <= 1.0e-12)
    {
        side = (std::abs(look.z) < 0.9) ? vsg::cross(look, vsg::dvec3(0.0, 0.0, 1.0))
                                        : vsg::cross(look, vsg::dvec3(1.0, 0.0, 0.0));
        const double len = vsg::length(side);
        if (len <= 1.0e-12) return;
        side /= len;
    }
    else
    {
        side /= sideLen;
    }
    _cubeLookAt->up = vsg::normalize(vsg::cross(side, look));
}

void ControlCube::updateInset()
{
    if (!_window || !_cubeViewport || !_cubeOrtho || !_depthClear) return;

    const VkExtent2D extent = _window->extent2D();
    if (extent.width == 0 || extent.height == 0) return;

    const double dpr = _qtWindow ? _qtWindow->devicePixelRatio() : 1.0;
    int32_t size = scaledPixels(insetSize, dpr);
    const int32_t margin = scaledPixels(insetMargin, dpr);

    const int32_t width = static_cast<int32_t>(extent.width);
    const int32_t height = static_cast<int32_t>(extent.height);
    if (margin + size > width) size = std::max(1, width - margin);
    if (margin + size > height) size = std::max(1, height - margin);

    _insetExtent = size;
    _insetX = margin;
    _insetY = height - margin - size;
    if (_insetY < 0) _insetY = 0;

    _cubeViewport->set(_insetX, _insetY, static_cast<uint32_t>(size), static_cast<uint32_t>(size));
    _cubeOrtho->left = -cubeOrthoHalf;
    _cubeOrtho->right = cubeOrthoHalf;
    _cubeOrtho->bottom = -cubeOrthoHalf;
    _cubeOrtho->top = cubeOrthoHalf;

    if (!_depthClear->rects.empty())
    {
        auto& rect = _depthClear->rects.front();
        rect.rect.offset = {_insetX, _insetY};
        rect.rect.extent = {static_cast<uint32_t>(size), static_cast<uint32_t>(size)};
    }
}

bool ControlCube::unproject(int32_t x, int32_t y, vsg::dvec3& origin, vsg::dvec3& direction) const
{
    if (!_cubeCamera || !_cubeCamera->projectionMatrix || !_cubeCamera->viewMatrix)
        return false;

    const VkViewport vp = _cubeCamera->getViewport();
    if (vp.width <= 0.0f || vp.height <= 0.0f) return false;

    const vsg::dvec3 ndcNear(
        ((static_cast<double>(x) - static_cast<double>(vp.x)) / static_cast<double>(vp.width)) * 2.0 - 1.0,
        ((static_cast<double>(y) - static_cast<double>(vp.y)) / static_cast<double>(vp.height)) * 2.0 - 1.0,
        static_cast<double>(vp.minDepth) * 2.0 - 1.0);
    const vsg::dvec3 ndcFar(ndcNear.x, ndcNear.y, static_cast<double>(vp.maxDepth) * 2.0 - 1.0);

    const vsg::dmat4 inv = vsg::inverse(_cubeCamera->projectionMatrix->transform() *
                                        _cubeCamera->viewMatrix->transform());
    origin = inv * ndcNear;
    const vsg::dvec3 farPoint = inv * ndcFar;
    direction = farPoint - origin;
    const double dirLen = vsg::length(direction);
    if (dirLen <= 0.0) return false;
    direction /= dirLen;
    return true;
}

vsg::dvec3 ControlCube::insetTbc(int32_t x, int32_t y) const
{
    if (!_cubeCamera) return {0.0, 0.0, 1.0};
    const VkViewport vp = _cubeCamera->getViewport();
    if (vp.width <= 0.0f || vp.height <= 0.0f) return {0.0, 0.0, 1.0};

    const double aspect = static_cast<double>(vp.width) / static_cast<double>(vp.height);
    vsg::dvec2 v(
        ((static_cast<double>(x) - static_cast<double>(vp.x)) / static_cast<double>(vp.width) * 2.0 - 1.0) * aspect,
        (static_cast<double>(y) - static_cast<double>(vp.y)) / static_cast<double>(vp.height) * 2.0 - 1.0);

    const double l = vsg::length(v);
    if (l < 1.0)
        return {v.x, -v.y, 0.5 + std::cos(l * pi) * 0.5};
    return {v.x, -v.y, 0.0};
}

void ControlCube::orbitDrag(int32_t x, int32_t y)
{
    if (!_trackball) return;

    const vsg::dvec3 newTbc = insetTbc(x, y);
    const double prevLen = vsg::length(_prevTbc);
    const double newLen = vsg::length(newTbc);
    if (prevLen <= 1.0e-12 || newLen <= 1.0e-12)
    {
        _prevTbc = newTbc;
        return;
    }

    vsg::dvec3 xp = vsg::cross(newTbc / newLen, _prevTbc / prevLen);
    const double xpLen = vsg::length(xp);
    if (xpLen > 0.0)
    {
        const double angle = std::asin(std::min(xpLen, 1.0));
        _trackball->rotate(angle, xp / xpLen);
    }
    _prevTbc = newTbc;
}

std::optional<vsg::dvec3> ControlCube::pickSnapDirection(int32_t x, int32_t y) const
{
    vsg::dvec3 origin;
    vsg::dvec3 dir;
    if (!unproject(x, y, origin, dir)) return std::nullopt;

    const double tMax = 1.0e6;
    double bestT = tMax;
    vsg::dvec3 snap;

    for (const auto& corner : cornerHandles())
    {
        double t = 0.0;
        if (intersectSphere(origin, dir, corner.center, handleSphereRadius, bestT, t))
        {
            bestT = t;
            snap = corner.snap;
        }
    }
    if (bestT < tMax) return snap;

    for (const auto& edge : edgeHandles())
    {
        double t = 0.0;
        if (intersectTube(origin, dir, edge, handleTubeRadius, bestT, t))
        {
            bestT = t;
            snap = edge.snap;
        }
    }
    if (bestT < tMax) return snap;

    double t = 0.0;
    if (intersectFace(origin, dir, tMax, snap, t))
        return snap;
    return std::nullopt;
}

void ControlCube::snapToDirection(const vsg::dvec3& direction)
{
    if (!_mainCamera || !_trackball) return;
    auto* mainLookAt = dynamic_cast<vsg::LookAt*>(_mainCamera->viewMatrix.get());
    if (!mainLookAt) return;

    vsg::dvec3 faceNormal = direction;
    const double nlen = vsg::length(faceNormal);
    if (nlen <= 1.0e-12) return;
    faceNormal /= nlen;

    const vsg::dvec3 center = mainLookAt->center;
    double dist = vsg::length(mainLookAt->eye - center);
    if (dist <= 1.0e-6) dist = 1.0;

    const vsg::dvec3 eye = center + faceNormal * dist;
    const vsg::dvec3 up = (std::abs(faceNormal.z) > 0.99) ? vsg::dvec3(0.0, 1.0, 0.0)
                                                          : vsg::dvec3(0.0, 0.0, 1.0);
    _trackball->setViewpoint(vsg::LookAt::create(eye, center, up), snapDuration);
}

} // namespace app
