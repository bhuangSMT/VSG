#include "StlImporter.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace app
{

namespace
{

// Read a little-endian 32-bit float from a byte buffer at the given offset.
float readFloatLE(const char* data)
{
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(float));
    return value;
}

std::uint32_t readUInt32LE(const char* data)
{
    std::uint32_t value = 0;
    std::memcpy(&value, data, sizeof(std::uint32_t));
    return value;
}

} // namespace

bool StlImporter::looksLikeBinary(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) return false;

    const std::streamoff fileSize = file.tellg();

    // Binary STL: 80-byte header + 4-byte count + 50 bytes per triangle.
    if (fileSize < 84) return false;

    file.seekg(80, std::ios::beg);
    char countBytes[4] = {};
    if (!file.read(countBytes, 4)) return false;

    const std::uint32_t triangleCount = readUInt32LE(countBytes);
    const std::streamoff expectedSize =
        std::streamoff(84) + std::streamoff(triangleCount) * std::streamoff(50);

    return fileSize == expectedSize;
}

TriangleMesh StlImporter::import(const std::string& filename) const
{
    if (looksLikeBinary(filename))
        return importBinary(filename);
    return importAscii(filename);
}

TriangleMesh StlImporter::importBinary(const std::string& filename) const
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open STL file: " + filename);

    char header[80] = {};
    if (!file.read(header, 80))
        throw std::runtime_error("STL file too short to contain a header: " + filename);

    char countBytes[4] = {};
    if (!file.read(countBytes, 4))
        throw std::runtime_error("STL file missing triangle count: " + filename);

    const std::uint32_t triangleCount = readUInt32LE(countBytes);

    TriangleMesh mesh;
    mesh.triangles.reserve(triangleCount);

    // Each facet: 12 floats (3 for the normal + 9 for the corners) + 2 pad bytes.
    char facet[50] = {};
    for (std::uint32_t i = 0; i < triangleCount; ++i)
    {
        if (!file.read(facet, 50))
            throw std::runtime_error("Unexpected end of binary STL file: " + filename);

        MeshTriangle tri;
        tri.normal.set(readFloatLE(facet + 0), readFloatLE(facet + 4), readFloatLE(facet + 8));
        tri.v0.set(readFloatLE(facet + 12), readFloatLE(facet + 16), readFloatLE(facet + 20));
        tri.v1.set(readFloatLE(facet + 24), readFloatLE(facet + 28), readFloatLE(facet + 32));
        tri.v2.set(readFloatLE(facet + 36), readFloatLE(facet + 40), readFloatLE(facet + 44));
        mesh.triangles.push_back(tri);
    }

    return mesh;
}

TriangleMesh StlImporter::importAscii(const std::string& filename) const
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Unable to open STL file: " + filename);

    TriangleMesh mesh;

    std::string token;
    auto readVec3 = [&file](vsg::vec3& out) -> bool {
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!(file >> x >> y >> z)) return false;
        out.set(x, y, z);
        return true;
    };

    MeshTriangle current;
    int cornerIndex = 0;

    while (file >> token)
    {
        if (token == "solid")
        {
            // Optional solid name runs to the end of the line.
            std::string rest;
            std::getline(file, rest);
            std::istringstream iss(rest);
            iss >> mesh.name;
        }
        else if (token == "facet")
        {
            std::string normalKeyword;
            file >> normalKeyword; // "normal"
            if (!readVec3(current.normal))
                throw std::runtime_error("Malformed facet normal in ASCII STL: " + filename);
            cornerIndex = 0;
        }
        else if (token == "vertex")
        {
            vsg::vec3 v;
            if (!readVec3(v))
                throw std::runtime_error("Malformed vertex in ASCII STL: " + filename);

            if (cornerIndex == 0) current.v0 = v;
            else if (cornerIndex == 1) current.v1 = v;
            else if (cornerIndex == 2) current.v2 = v;
            ++cornerIndex;
        }
        else if (token == "endfacet")
        {
            if (cornerIndex != 3)
                throw std::runtime_error("Facet with wrong vertex count in ASCII STL: " + filename);
            mesh.triangles.push_back(current);
        }
        // "outer", "loop", "endloop", "endsolid" are ignored.
    }

    if (mesh.triangles.empty())
        throw std::runtime_error("No triangles found in STL file: " + filename);

    return mesh;
}

} // namespace app
