#include "BRep.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace app
{

namespace
{

// Integer-quantised position used as a hash key so that coincident (or nearly
// coincident) STL corners are welded into a single topological vertex.
struct QuantizedKey
{
    std::int64_t x = 0;
    std::int64_t y = 0;
    std::int64_t z = 0;

    bool operator==(const QuantizedKey& rhs) const
    {
        return x == rhs.x && y == rhs.y && z == rhs.z;
    }
};

struct QuantizedKeyHash
{
    std::size_t operator()(const QuantizedKey& k) const noexcept
    {
        // Simple hash combine.
        std::size_t h = std::hash<std::int64_t>{}(k.x);
        h ^= std::hash<std::int64_t>{}(k.y) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<std::int64_t>{}(k.z) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

QuantizedKey quantize(const vsg::vec3& p, double invTolerance)
{
    QuantizedKey k;
    k.x = std::llround(static_cast<double>(p.x) * invTolerance);
    k.y = std::llround(static_cast<double>(p.y) * invTolerance);
    k.z = std::llround(static_cast<double>(p.z) * invTolerance);
    return k;
}

// Undirected edge key packing two 32-bit vertex ids (smaller id first).
std::uint64_t edgeKey(std::uint32_t a, std::uint32_t b)
{
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
}

} // namespace

BRep BRep::fromTriangles(const TriangleMesh& mesh)
{
    BRep brep;

    // Determine a welding tolerance from the model's bounding box so it scales
    // with the geometry rather than assuming absolute units.
    vsg::vec3 bbMin(std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max());
    vsg::vec3 bbMax(std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest());

    auto expand = [&bbMin, &bbMax](const vsg::vec3& v) {
        bbMin.x = std::min(bbMin.x, v.x);
        bbMin.y = std::min(bbMin.y, v.y);
        bbMin.z = std::min(bbMin.z, v.z);
        bbMax.x = std::max(bbMax.x, v.x);
        bbMax.y = std::max(bbMax.y, v.y);
        bbMax.z = std::max(bbMax.z, v.z);
    };

    for (const auto& tri : mesh.triangles)
    {
        expand(tri.v0);
        expand(tri.v1);
        expand(tri.v2);
    }

    double diagonal = 0.0;
    if (!mesh.triangles.empty())
    {
        const vsg::vec3 extent = bbMax - bbMin;
        diagonal = std::sqrt(static_cast<double>(extent.x) * extent.x +
                             static_cast<double>(extent.y) * extent.y +
                             static_cast<double>(extent.z) * extent.z);
    }

    const double tolerance = std::max(diagonal * 1e-6, 1e-9);
    const double invTolerance = 1.0 / tolerance;

    std::unordered_map<QuantizedKey, std::uint32_t, QuantizedKeyHash> vertexLookup;
    vertexLookup.reserve(mesh.triangles.size() * 3);

    auto weld = [&](const vsg::vec3& p) -> std::uint32_t {
        const QuantizedKey key = quantize(p, invTolerance);
        auto it = vertexLookup.find(key);
        if (it != vertexLookup.end()) return it->second;

        const std::uint32_t id = static_cast<std::uint32_t>(brep._vertices.size());
        brep._vertices.push_back(p);
        vertexLookup.emplace(key, id);
        return id;
    };

    brep._faceOffsets.push_back(0);
    brep._faceVertices.reserve(mesh.triangles.size() * 3);

    for (const auto& tri : mesh.triangles)
    {
        const std::uint32_t a = weld(tri.v0);
        const std::uint32_t b = weld(tri.v1);
        const std::uint32_t c = weld(tri.v2);

        // Drop degenerate triangles (zero area after welding) so they do not
        // create spurious edges during validation.
        if (a == b || b == c || a == c)
        {
            ++brep._degenerateFaceCount;
            continue;
        }

        brep._faceVertices.push_back(a);
        brep._faceVertices.push_back(b);
        brep._faceVertices.push_back(c);
        brep._faceOffsets.push_back(static_cast<std::uint32_t>(brep._faceVertices.size()));
    }

    const std::size_t inputCorners = mesh.triangles.size() * 3;
    if (inputCorners >= brep._vertices.size())
        brep._weldedVertexCount = inputCorners - brep._vertices.size();

    // Built last, over the finished topology.
    brep._bvh = BVH::build(brep);

    return brep;
}

BRep::ValidationResult BRep::validate() const
{
    ValidationResult result;
    result.degenerateFaceCount = _degenerateFaceCount;
    result.weldedVertexCount = _weldedVertexCount;

    // Count how many faces reference each undirected edge.
    std::unordered_map<std::uint64_t, int> edgeUse;
    edgeUse.reserve(_faceVertices.size());

    const std::size_t faces = faceCount();
    for (std::size_t f = 0; f < faces; ++f)
    {
        const std::uint32_t begin = _faceOffsets[f];
        const std::uint32_t end = _faceOffsets[f + 1];
        const std::uint32_t count = end - begin;
        if (count < 3) continue;

        for (std::uint32_t k = 0; k < count; ++k)
        {
            const std::uint32_t a = _faceVertices[begin + k];
            const std::uint32_t b = _faceVertices[begin + ((k + 1) % count)];
            if (a == b) continue;
            ++edgeUse[edgeKey(a, b)];
        }
    }

    for (const auto& entry : edgeUse)
    {
        if (entry.second == 1)
            ++result.boundaryEdgeCount;
        else if (entry.second > 2)
            ++result.nonManifoldEdgeCount;
    }

    result.openEdgeCount = result.boundaryEdgeCount + result.nonManifoldEdgeCount;
    result.watertight = (faces > 0) && (result.openEdgeCount == 0);

    return result;
}

std::vector<std::uint32_t> BRep::extractEdgeIndices() const
{
    std::vector<std::uint32_t> edges;
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(_faceVertices.size());

    const std::size_t faces = faceCount();
    for (std::size_t f = 0; f < faces; ++f)
    {
        const std::uint32_t begin = _faceOffsets[f];
        const std::uint32_t end = _faceOffsets[f + 1];
        const std::uint32_t count = end - begin;
        if (count < 2) continue;

        for (std::uint32_t k = 0; k < count; ++k)
        {
            const std::uint32_t a = _faceVertices[begin + k];
            const std::uint32_t b = _faceVertices[begin + ((k + 1) % count)];
            if (a == b) continue;

            if (seen.insert(edgeKey(a, b)).second)
            {
                edges.push_back(a);
                edges.push_back(b);
            }
        }
    }

    return edges;
}

} // namespace app
