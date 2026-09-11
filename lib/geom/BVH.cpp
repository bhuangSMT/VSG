#include "BVH.h"

#include <algorithm>
#include <limits>

#include "BRep.h"
#include "TriangleMesh.h"

namespace app
{

namespace
{

// Small enough that the leaf scan stays short, large enough that the tree does
// not cost more to walk than the triangles it saves.
constexpr std::uint32_t maxLeafFaces = 4;

} // namespace

BoundingBox BVH::bounds() const
{
    if (_nodes.empty()) return {};
    const Node& root = _nodes.front();
    return BoundingBox(Point3d{static_cast<double>(root.min[0]),
                               static_cast<double>(root.min[1]),
                               static_cast<double>(root.min[2])},
                       Point3d{static_cast<double>(root.max[0]),
                               static_cast<double>(root.max[1]),
                               static_cast<double>(root.max[2])});
}

BVH BVH::build(const BRep& brep)
{
    BVH bvh;

    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();

    const std::size_t faces = brep.faceCount();
    if (faces == 0) return bvh;

    std::vector<FaceBounds> bounds(faces);
    bvh._faces.reserve(faces);

    for (std::size_t f = 0; f < faces; ++f)
    {
        const std::uint32_t begin = faceOffsets[f];
        const std::uint32_t end = faceOffsets[f + 1];
        if (end - begin < 3) continue;

        FaceBounds& fb = bounds[f];
        const vsg::vec3& first = verts[faceVertices[begin]];
        for (int a = 0; a < 3; ++a) fb.min[a] = fb.max[a] = first[a];

        for (std::uint32_t k = begin + 1; k < end; ++k)
        {
            const vsg::vec3& p = verts[faceVertices[k]];
            for (int a = 0; a < 3; ++a)
            {
                fb.min[a] = std::min(fb.min[a], p[a]);
                fb.max[a] = std::max(fb.max[a], p[a]);
            }
        }

        for (int a = 0; a < 3; ++a) fb.centroid[a] = 0.5f * (fb.min[a] + fb.max[a]);

        bvh._faces.push_back(static_cast<std::uint32_t>(f));
    }

    if (bvh._faces.empty()) return bvh;

    bvh._nodes.emplace_back();
    bvh.buildNode(0, bounds, 0, static_cast<std::uint32_t>(bvh._faces.size()), 0);

    return bvh;
}

BVH BVH::build(const TriangleMesh& mesh)
{
    BVH bvh;

    const std::size_t faces = mesh.triangles.size();
    if (faces == 0) return bvh;

    std::vector<FaceBounds> bounds(faces);
    bvh._faces.reserve(faces);

    for (std::size_t f = 0; f < faces; ++f)
    {
        const MeshTriangle& tri = mesh.triangles[f];
        const vsg::vec3 corners[3]{tri.v0, tri.v1, tri.v2};

        FaceBounds& fb = bounds[f];
        for (int a = 0; a < 3; ++a) fb.min[a] = fb.max[a] = corners[0][a];

        for (int c = 1; c < 3; ++c)
        {
            for (int a = 0; a < 3; ++a)
            {
                fb.min[a] = std::min(fb.min[a], corners[c][a]);
                fb.max[a] = std::max(fb.max[a], corners[c][a]);
            }
        }

        for (int a = 0; a < 3; ++a) fb.centroid[a] = 0.5f * (fb.min[a] + fb.max[a]);

        bvh._faces.push_back(static_cast<std::uint32_t>(f));
    }

    if (bvh._faces.empty()) return bvh;

    bvh._nodes.emplace_back();
    bvh.buildNode(0, bounds, 0, static_cast<std::uint32_t>(bvh._faces.size()), 0);

    return bvh;
}

void BVH::buildNode(std::uint32_t self, const std::vector<FaceBounds>& bounds,
                    std::uint32_t begin, std::uint32_t end, int depth)
{
    constexpr float inf = std::numeric_limits<float>::infinity();

    float lo[3]{inf, inf, inf};
    float hi[3]{-inf, -inf, -inf};
    float centroidLo[3]{inf, inf, inf};
    float centroidHi[3]{-inf, -inf, -inf};

    for (std::uint32_t i = begin; i < end; ++i)
    {
        const FaceBounds& fb = bounds[_faces[i]];
        for (int a = 0; a < 3; ++a)
        {
            lo[a] = std::min(lo[a], fb.min[a]);
            hi[a] = std::max(hi[a], fb.max[a]);
            centroidLo[a] = std::min(centroidLo[a], fb.centroid[a]);
            centroidHi[a] = std::max(centroidHi[a], fb.centroid[a]);
        }
    }

    // Written through the index rather than a reference: the recursion below
    // appends to _nodes, which can move the whole array.
    for (int a = 0; a < 3; ++a)
    {
        _nodes[self].min[a] = lo[a];
        _nodes[self].max[a] = hi[a];
    }

    const std::uint32_t count = end - begin;
    if (count <= maxLeafFaces || depth >= maxDepth)
    {
        _nodes[self].firstChildOrFace = begin;
        _nodes[self].faceCount = count;
        return;
    }

    // Split on the axis the centroids are most spread out along, at their
    // midpoint. Cheaper than a surface-area sweep and good enough here, where
    // the triangles of an imported mesh are fairly evenly sized.
    int axis = 0;
    float widest = centroidHi[0] - centroidLo[0];
    for (int a = 1; a < 3; ++a)
    {
        const float width = centroidHi[a] - centroidLo[a];
        if (width > widest)
        {
            widest = width;
            axis = a;
        }
    }

    std::uint32_t mid = begin;
    if (widest > 0.0f)
    {
        const float split = 0.5f * (centroidLo[axis] + centroidHi[axis]);
        const auto pivot = std::partition(
            _faces.begin() + begin, _faces.begin() + end,
            [&bounds, axis, split](std::uint32_t f) { return bounds[f].centroid[axis] < split; });
        mid = static_cast<std::uint32_t>(pivot - _faces.begin());
    }

    // Centroids piled on one spot, or all of them on one side of the midpoint:
    // halve the range instead so the recursion still makes progress.
    if (mid == begin || mid == end) mid = begin + count / 2;

    const std::uint32_t firstChild = static_cast<std::uint32_t>(_nodes.size());
    _nodes.emplace_back();
    _nodes.emplace_back();

    _nodes[self].firstChildOrFace = firstChild;
    _nodes[self].faceCount = 0;

    buildNode(firstChild, bounds, begin, mid, depth + 1);
    buildNode(firstChild + 1, bounds, mid, end, depth + 1);
}

} // namespace app
