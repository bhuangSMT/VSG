#include "ThreeMfImporter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <zlib.h>

#include <QtCore/QByteArray>
#include <QtCore/QLatin1String>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QXmlStreamAttributes>
#include <QtCore/QXmlStreamReader>

namespace app
{

namespace
{

using Bytes = std::vector<unsigned char>;

// Everything in a ZIP is little endian.
std::uint16_t readU16(const unsigned char* at)
{
    return static_cast<std::uint16_t>(at[0] | (at[1] << 8));
}

std::uint32_t readU32(const unsigned char* at)
{
    return static_cast<std::uint32_t>(at[0]) | (static_cast<std::uint32_t>(at[1]) << 8) |
           (static_cast<std::uint32_t>(at[2]) << 16) | (static_cast<std::uint32_t>(at[3]) << 24);
}

Bytes readWholeFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open 3MF file: " + filename);

    const std::streamoff length = file.tellg();
    if (length < 0) throw std::runtime_error("Cannot determine the size of: " + filename);

    file.seekg(0, std::ios::beg);

    // The length came from the stream itself, so it is a real file size rather
    // than anything derived from the file's own contents.
    Bytes bytes(static_cast<Bytes::size_type>(length));
    if (length > 0 && !file.read(reinterpret_cast<char*>(bytes.data()), length))
        throw std::runtime_error("Cannot read 3MF file: " + filename);

    return bytes;
}

// ---------------------------------------------------------------------------
// ZIP reading
//
// Only as much of the format as an OPC package needs: walk the central
// directory, then inflate a named part. Anything beyond that is refused with a
// message rather than guessed at.
// ---------------------------------------------------------------------------

constexpr std::uint32_t endOfCentralDirectorySignature = 0x06054b50;
constexpr std::uint32_t centralFileHeaderSignature = 0x02014b50;
constexpr std::uint32_t localFileHeaderSignature = 0x04034b50;

constexpr std::int64_t endOfCentralDirectorySize = 22;
constexpr std::int64_t centralFileHeaderSize = 46;
constexpr std::int64_t localFileHeaderSize = 30;

// Parts are held in memory whole, so this caps how much one is allowed to
// claim. It also keeps the sizes inside what zlib's unsigned int counters and a
// single allocation can take.
constexpr std::int64_t maxPartSize = 1LL << 30;

struct ZipEntry
{
    std::int64_t localHeaderOffset = 0;
    std::int64_t compressedSize = 0;
    std::int64_t uncompressedSize = 0;
    std::uint16_t method = 0;
    std::uint16_t flags = 0;
};

std::int64_t findEndOfCentralDirectory(const Bytes& zip)
{
    const std::int64_t size = static_cast<std::int64_t>(zip.size());
    if (size < endOfCentralDirectorySize)
        throw std::runtime_error("The file is too small to be a 3MF package.");

    // The record is last except for an optional trailing comment, whose length
    // is a 16-bit field, so this is as far back as it can possibly start.
    const std::int64_t earliest =
        std::max<std::int64_t>(0, size - endOfCentralDirectorySize - 0xffff);

    for (std::int64_t at = size - endOfCentralDirectorySize; at >= earliest; --at)
    {
        if (readU32(zip.data() + at) == endOfCentralDirectorySignature) return at;
    }

    throw std::runtime_error("Not a 3MF package: no ZIP end of central directory record.");
}

std::map<std::string, ZipEntry> readCentralDirectory(const Bytes& zip)
{
    const std::int64_t size = static_cast<std::int64_t>(zip.size());
    const unsigned char* record = zip.data() + findEndOfCentralDirectory(zip);

    const std::uint16_t entryCount = readU16(record + 10);
    const std::uint32_t directorySize = readU32(record + 12);
    const std::uint32_t directoryOffset = readU32(record + 16);

    // The all-ones sentinels mean the real values live in a ZIP64 record.
    if (entryCount == 0xffff || directorySize == 0xffffffffu || directoryOffset == 0xffffffffu)
        throw std::runtime_error("ZIP64 3MF packages are not supported.");

    std::int64_t at = directoryOffset;
    if (at < 0 || at + directorySize > size)
        throw std::runtime_error("Corrupt 3MF package: the central directory is out of bounds.");

    std::map<std::string, ZipEntry> entries;
    for (std::uint16_t i = 0; i < entryCount; ++i)
    {
        if (at + centralFileHeaderSize > size ||
            readU32(zip.data() + at) != centralFileHeaderSignature)
            throw std::runtime_error("Corrupt 3MF package: bad central directory entry.");

        const unsigned char* header = zip.data() + at;

        ZipEntry entry;
        entry.flags = readU16(header + 8);
        entry.method = readU16(header + 10);
        entry.compressedSize = readU32(header + 20);
        entry.uncompressedSize = readU32(header + 24);
        entry.localHeaderOffset = readU32(header + 42);

        const std::int64_t nameLength = readU16(header + 28);
        const std::int64_t extraLength = readU16(header + 30);
        const std::int64_t commentLength = readU16(header + 32);

        if (at + centralFileHeaderSize + nameLength > size)
            throw std::runtime_error("Corrupt 3MF package: truncated entry name.");

        std::string name(reinterpret_cast<const char*>(header + centralFileHeaderSize),
                         static_cast<std::string::size_type>(nameLength));

        entries.emplace(std::move(name), entry);

        at += centralFileHeaderSize + nameLength + extraLength + commentLength;
    }

    return entries;
}

Bytes inflateRaw(const unsigned char* data, std::int64_t compressedSize,
                 std::int64_t uncompressedSize)
{
    // Both counts were bounded against maxPartSize by the caller, and zlib
    // takes them as unsigned int, so narrowing here cannot lose anything.
    Bytes out(static_cast<Bytes::size_type>(uncompressedSize));

    z_stream stream{};
    // A negative window size selects a raw deflate stream, which is what a ZIP
    // entry holds: there is no zlib header in front of it.
    if (inflateInit2(&stream, -MAX_WBITS) != Z_OK)
        throw std::runtime_error("Cannot start decompression of the 3MF package.");

    stream.next_in = const_cast<Bytef*>(data);
    stream.avail_in = static_cast<uInt>(compressedSize);
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(uncompressedSize);

    const int result = inflate(&stream, Z_FINISH);
    inflateEnd(&stream);

    if (result != Z_STREAM_END)
        throw std::runtime_error("The 3MF package is corrupt: decompression failed.");

    return out;
}

Bytes extractEntry(const Bytes& zip, const std::string& name, const ZipEntry& entry)
{
    const std::int64_t size = static_cast<std::int64_t>(zip.size());

    if ((entry.flags & 0x0001) != 0)
        throw std::runtime_error("Encrypted 3MF packages are not supported.");

    if (entry.uncompressedSize < 0 || entry.uncompressedSize > maxPartSize ||
        entry.compressedSize < 0 || entry.compressedSize > maxPartSize)
        throw std::runtime_error("The 3MF part '" + name + "' is too large to read.");

    std::int64_t at = entry.localHeaderOffset;
    if (at < 0 || at + localFileHeaderSize > size ||
        readU32(zip.data() + at) != localFileHeaderSignature)
        throw std::runtime_error("Corrupt 3MF package: bad local header for '" + name + "'.");

    // The local header's own sizes are unreliable — they are left as zero when
    // the writer puts them in a trailing data descriptor — so the central
    // directory's copies are the ones used. Only the variable-length fields are
    // read from here.
    const std::int64_t nameLength = readU16(zip.data() + at + 26);
    const std::int64_t extraLength = readU16(zip.data() + at + 28);

    at += localFileHeaderSize + nameLength + extraLength;
    if (at < 0 || at + entry.compressedSize > size)
        throw std::runtime_error("Corrupt 3MF package: '" + name + "' runs past the end.");

    const unsigned char* data = zip.data() + at;

    if (entry.method == 0)
    {
        if (entry.compressedSize != entry.uncompressedSize)
            throw std::runtime_error("Corrupt 3MF package: '" + name + "' has inconsistent sizes.");
        return Bytes(data, data + entry.compressedSize);
    }

    if (entry.method == 8) return inflateRaw(data, entry.compressedSize, entry.uncompressedSize);

    throw std::runtime_error("The 3MF part '" + name + "' uses an unsupported compression method.");
}

// ---------------------------------------------------------------------------
// Model XML
// ---------------------------------------------------------------------------

// 3MF writes a transform as twelve numbers: three rows of the linear part
// followed by the translation. Points are row vectors, so v' = v * M and the
// implied fourth column is (0, 0, 0, 1).
struct Matrix
{
    double m[12]{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
};

vsg::vec3 transformPoint(const Matrix& t, double x, double y, double z)
{
    return vsg::vec3(static_cast<float>(x * t.m[0] + y * t.m[3] + z * t.m[6] + t.m[9]),
                     static_cast<float>(x * t.m[1] + y * t.m[4] + z * t.m[7] + t.m[10]),
                     static_cast<float>(x * t.m[2] + y * t.m[5] + z * t.m[8] + t.m[11]));
}

// The single matrix equivalent to applying `first` and then `second`.
Matrix compose(const Matrix& first, const Matrix& second)
{
    Matrix result;

    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
        {
            result.m[row * 3 + column] = first.m[row * 3 + 0] * second.m[column] +
                                         first.m[row * 3 + 1] * second.m[3 + column] +
                                         first.m[row * 3 + 2] * second.m[6 + column];
        }
    }

    // The translation row carries an implied w of 1, so it picks up the second
    // transform's translation as well.
    for (int column = 0; column < 3; ++column)
    {
        result.m[9 + column] = first.m[9] * second.m[column] + first.m[10] * second.m[3 + column] +
                               first.m[11] * second.m[6 + column] + second.m[9 + column];
    }

    return result;
}

Matrix scaleMatrix(double factor)
{
    Matrix t;
    t.m[0] = t.m[4] = t.m[8] = factor;
    return t;
}

Matrix parseMatrix(const QString& text)
{
    const QStringList parts = text.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (parts.size() != 12)
        throw std::runtime_error("The 3MF file has a transform that is not twelve numbers.");

    Matrix t;
    for (int i = 0; i < 12; ++i)
    {
        bool ok = false;
        t.m[i] = parts[i].toDouble(&ok);
        if (!ok) throw std::runtime_error("The 3MF file has a transform that is not numeric.");
    }
    return t;
}

// Everything is brought to millimetres, which is also the format's default, so
// that a 3MF and an STL of the same part arrive at the same size.
double unitToMillimetres(const QString& unit)
{
    if (unit.isEmpty() || unit == QLatin1String("millimeter")) return 1.0;
    if (unit == QLatin1String("micron")) return 0.001;
    if (unit == QLatin1String("centimeter")) return 10.0;
    if (unit == QLatin1String("inch")) return 25.4;
    if (unit == QLatin1String("foot")) return 304.8;
    if (unit == QLatin1String("meter")) return 1000.0;

    throw std::runtime_error("The 3MF file declares an unknown unit: " + unit.toStdString());
}

// One object reference, from either a build item or a component. An empty path
// means the same part the reference was written in; a non-empty one is the
// production extension pointing at another part of the package.
struct Reference
{
    std::string path;
    int id = 0;
    Matrix transform;
};

struct Object
{
    std::vector<double> vertices;  // x, y, z triples
    std::vector<int> triangles;    // index triples
    std::vector<Reference> components;
};

// One model part. Object ids are numbered per part, not per package, so they
// are only meaningful together with the path the part was read from.
struct Part
{
    std::map<int, Object> objects;
    std::vector<Reference> build;
    double unitScale = 1.0;
};

// Attributes are matched on local name so that a prefix does not hide them.
// The production extension's path attribute in particular is always prefixed
// (p:path), and writers are free to choose the prefix.
using AttributeValue = decltype(std::declval<QXmlStreamAttribute>().value());

AttributeValue localValue(const QXmlStreamAttributes& attributes, const char* name)
{
    for (const QXmlStreamAttribute& attribute : attributes)
    {
        if (attribute.name() == QLatin1String(name)) return attribute.value();
    }
    return AttributeValue{};
}

int requireInt(const QXmlStreamAttributes& attributes, const char* name, const char* element)
{
    bool ok = false;
    const int value = localValue(attributes, name).toInt(&ok);
    if (!ok)
        throw std::runtime_error(std::string("The 3MF file has a <") + element + "> without a '" +
                                 name + "' attribute.");
    return value;
}

double requireDouble(const QXmlStreamAttributes& attributes, const char* name, const char* element)
{
    bool ok = false;
    const double value = localValue(attributes, name).toDouble(&ok);
    if (!ok)
        throw std::runtime_error(std::string("The 3MF file has a <") + element + "> without a '" +
                                 name + "' attribute.");
    return value;
}

// Part paths are written absolute within the package ("/3D/Objects/x.model"),
// while ZIP entries are named relative, so the leading slash comes off.
std::string normalisePartPath(const QString& path)
{
    std::string result = path.toStdString();
    if (!result.empty() && result.front() == '/') result.erase(result.begin());
    return result;
}

Reference readReference(const QXmlStreamAttributes& attributes, const char* element)
{
    Reference reference;
    reference.id = requireInt(attributes, "objectid", element);

    const AttributeValue transform = localValue(attributes, "transform");
    if (!transform.isEmpty()) reference.transform = parseMatrix(transform.toString());

    const AttributeValue path = localValue(attributes, "path");
    if (!path.isEmpty()) reference.path = normalisePartPath(path.toString());

    return reference;
}

Part parsePart(const Bytes& xml)
{
    QXmlStreamReader reader(
        QByteArray(reinterpret_cast<const char*>(xml.data()), static_cast<int>(xml.size())));

    Part part;

    // Element names are unique across the core spec, so the ones that matter
    // can be picked out without tracking the full path. The only state needed
    // is which <object> is being filled in.
    Object* current = nullptr;

    while (!reader.atEnd())
    {
        reader.readNext();

        if (reader.isEndElement() && reader.name() == QLatin1String("object"))
        {
            current = nullptr;
            continue;
        }

        if (!reader.isStartElement()) continue;

        const QXmlStreamAttributes attributes = reader.attributes();

        if (reader.name() == QLatin1String("model"))
        {
            part.unitScale = unitToMillimetres(localValue(attributes, "unit").toString());
        }
        else if (reader.name() == QLatin1String("object"))
        {
            current = &part.objects[requireInt(attributes, "id", "object")];
        }
        else if (reader.name() == QLatin1String("vertex"))
        {
            if (!current) throw std::runtime_error("The 3MF file has a <vertex> outside an object.");
            current->vertices.push_back(requireDouble(attributes, "x", "vertex"));
            current->vertices.push_back(requireDouble(attributes, "y", "vertex"));
            current->vertices.push_back(requireDouble(attributes, "z", "vertex"));
        }
        else if (reader.name() == QLatin1String("triangle"))
        {
            if (!current)
                throw std::runtime_error("The 3MF file has a <triangle> outside an object.");
            current->triangles.push_back(requireInt(attributes, "v1", "triangle"));
            current->triangles.push_back(requireInt(attributes, "v2", "triangle"));
            current->triangles.push_back(requireInt(attributes, "v3", "triangle"));
        }
        else if (reader.name() == QLatin1String("component"))
        {
            if (!current)
                throw std::runtime_error("The 3MF file has a <component> outside an object.");
            current->components.push_back(readReference(attributes, "component"));
        }
        else if (reader.name() == QLatin1String("item"))
        {
            part.build.push_back(readReference(attributes, "item"));
        }
    }

    if (reader.hasError())
        throw std::runtime_error("A 3MF model part is not valid XML: " +
                                 reader.errorString().toStdString());

    return part;
}

// The package, with its model parts parsed on demand. Slicers routinely write
// one part per object and reference them from the root, so the parts that are
// actually built are the only ones read.
class Package
{
public:
    explicit Package(Bytes zip) : _zip(std::move(zip)), _entries(readCentralDirectory(_zip)) {}

    // The part named by the package relationships, falling back to the
    // conventional location and then to whatever model part exists.
    std::string rootPath() const;

    const Part& part(const std::string& path);

private:
    Bytes _zip;
    std::map<std::string, ZipEntry> _entries;
    std::map<std::string, Part> _parts;
};

std::string Package::rootPath() const
{
    const auto relationships = _entries.find("_rels/.rels");
    if (relationships != _entries.end())
    {
        const Bytes xml = extractEntry(_zip, relationships->first, relationships->second);
        QXmlStreamReader reader(
            QByteArray(reinterpret_cast<const char*>(xml.data()), static_cast<int>(xml.size())));

        while (!reader.atEnd())
        {
            reader.readNext();
            if (!reader.isStartElement() || reader.name() != QLatin1String("Relationship")) continue;

            const QXmlStreamAttributes attributes = reader.attributes();
            if (!localValue(attributes, "Type").endsWith(QLatin1String("/3dmodel"))) continue;

            const std::string target = normalisePartPath(localValue(attributes, "Target").toString());
            if (_entries.count(target) != 0) return target;
        }
    }

    if (_entries.count("3D/3dmodel.model") != 0) return "3D/3dmodel.model";

    const std::string suffix = ".model";
    for (const auto& entry : _entries)
    {
        const std::string& name = entry.first;
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            return name;
    }

    throw std::runtime_error("The 3MF package contains no model part.");
}

const Part& Package::part(const std::string& path)
{
    const auto parsed = _parts.find(path);
    if (parsed != _parts.end()) return parsed->second;

    const auto entry = _entries.find(path);
    if (entry == _entries.end())
        throw std::runtime_error("The 3MF package is missing the part '" + path + "'.");

    return _parts.emplace(path, parsePart(extractEntry(_zip, path, entry->second))).first->second;
}

// Components can nest, and nothing in the file stops them referring to one
// another in a loop, so the depth is capped rather than trusted.
constexpr int maxComponentDepth = 32;

void emitObject(Package& package, const std::string& partPath, int id, const Matrix& transform,
                int depth, TriangleMesh& mesh)
{
    if (depth > maxComponentDepth)
        throw std::runtime_error("The 3MF file nests components too deeply, or cyclically.");

    const Part& part = package.part(partPath);

    const auto found = part.objects.find(id);
    if (found == part.objects.end())
        throw std::runtime_error("The 3MF file references object " + std::to_string(id) +
                                 " in '" + partPath + "', which that part does not define.");

    const Object& object = found->second;

    const std::int64_t vertexCount = static_cast<std::int64_t>(object.vertices.size()) / 3;
    const std::int64_t triangleCount = static_cast<std::int64_t>(object.triangles.size()) / 3;

    for (std::int64_t t = 0; t < triangleCount; ++t)
    {
        const int* corner = object.triangles.data() + t * 3;
        if (corner[0] < 0 || corner[1] < 0 || corner[2] < 0 || corner[0] >= vertexCount ||
            corner[1] >= vertexCount || corner[2] >= vertexCount)
            throw std::runtime_error("The 3MF file has a triangle with an out-of-range vertex.");

        MeshTriangle triangle;

        vsg::vec3* corners[3] = {&triangle.v0, &triangle.v1, &triangle.v2};
        for (int k = 0; k < 3; ++k)
        {
            const double* v = object.vertices.data() + corner[k] * 3;
            *corners[k] = transformPoint(transform, v[0], v[1], v[2]);
        }

        // 3MF does not store face normals, so this comes from the winding, which
        // the spec requires to be counter-clockwise seen from outside.
        const vsg::vec3 normal = vsg::cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0);
        const float length = vsg::length(normal);
        triangle.normal = (length > 0.0f) ? normal / length : vsg::vec3(0.0f, 0.0f, 0.0f);

        mesh.triangles.push_back(triangle);
    }

    // A component without a path of its own lives in the same part as the
    // object that referenced it.
    for (const Reference& component : object.components)
    {
        emitObject(package, component.path.empty() ? partPath : component.path, component.id,
                   compose(component.transform, transform), depth + 1, mesh);
    }
}

} // namespace

TriangleMesh ThreeMfImporter::import(const std::string& filename) const
{
    Package package(readWholeFile(filename));

    const std::string rootPath = package.rootPath();
    const Part& root = package.part(rootPath);

    TriangleMesh mesh;
    mesh.name = filename;

    // The root part's unit governs the whole package.
    const Matrix unit = scaleMatrix(root.unitScale);

    if (!root.build.empty())
    {
        for (const Reference& item : root.build)
        {
            emitObject(package, item.path.empty() ? rootPath : item.path, item.id,
                       compose(item.transform, unit), 0, mesh);
        }
    }
    else
    {
        // A file with no build items is not strictly valid, but showing the
        // geometry it does define is more use than refusing it.
        for (const auto& entry : root.objects)
        {
            if (!entry.second.triangles.empty()) emitObject(package, rootPath, entry.first, unit, 0, mesh);
        }
    }

    if (mesh.triangles.empty()) throw std::runtime_error("The 3MF file contains no triangles.");

    return mesh;
}

} // namespace app
