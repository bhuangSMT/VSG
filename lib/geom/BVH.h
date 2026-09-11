// BVH - a bounding volume hierarchy over a BRep's faces.
//
// Built once when the BRep is built and then only read, so casting rays does
// not have to walk every triangle. Because the rays are axis aligned, a query
// is an infinite line along one axis: it ignores that axis entirely and reduces
// to a point-in-rectangle test against each node's footprint in the other two.
//
// Node bounds are floats taken from the vertex positions, expanded by one
// float ULP on query so a double sample sitting on a shared edge is not
// dropped by the AABB test while the exact triangle test would accept it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

#include "BoundingBox.h"

namespace app
{

class BRep;
struct TriangleMesh;

class BVH
{
public:
    // Build over every face of the BRep. Faces with fewer than three corners
    // are left out, matching what the ray caster is willing to intersect.
    static BVH build(const BRep& brep);

    // Build over every triangle of an unwelded soup. Face ids match the
    // triangle indices in mesh.triangles, so a visitor can index the soup
    // directly.
    static BVH build(const TriangleMesh& mesh);

    bool empty() const { return _nodes.empty(); }
    std::size_t nodeCount() const { return _nodes.size(); }

    // Axis-aligned bounds of the whole hierarchy (the root node), or an empty
    // box when the tree has not been built.
    BoundingBox bounds() const;

    // Call visit(face) for every face whose bounds contain (u0, v0) in the
    // plane spanned by axes u and v. Faces come back in no particular order and
    // the visitor still has to run the exact triangle test: this only narrows
    // the candidates down.
    template <typename Visitor>
    void query(std::size_t u, std::size_t v, double u0, double v0, Visitor&& visit) const;

    // Call visit(face) for every face whose AABB the ray origin + t * direction
    // can meet for some t in [tMin, tMax]. invDir is 1/direction per component
    // (with a large finite stand-in for zeros). Same contract as query(): the
    // visitor still has to run the exact triangle test.
    template <typename Visitor>
    void queryRay(const double origin[3], const double invDir[3],
                  double tMin, double tMax, Visitor&& visit) const;

private:
    struct Node
    {
        float min[3]{};
        float max[3]{};

        // An interior node stores the index of its first child, with the second
        // sitting right after it, and a face count of 0. A leaf stores the
        // offset of its first face in _faces and how many it owns.
        std::uint32_t firstChildOrFace = 0;
        std::uint32_t faceCount = 0;
    };

    // Bounds and centroid of one face, computed up front so that the build only
    // ever permutes indices rather than re-reading vertices.
    struct FaceBounds
    {
        float min[3]{};
        float max[3]{};
        float centroid[3]{};
    };

    // Fill node `self` from faces [begin, end) of _faces, recursing until the
    // range is small enough to be a leaf. Children are allocated as an adjacent
    // pair so a node only has to record the index of the first.
    void buildNode(std::uint32_t self, const std::vector<FaceBounds>& bounds,
                   std::uint32_t begin, std::uint32_t end, int depth);

    // Deep enough for any range that fits in _faces, given that a split which
    // fails to separate the centroids falls back to halving the range.
    static constexpr int maxDepth = 48;

    std::vector<Node> _nodes;

    // Face ids, permuted during the build so that each leaf owns a contiguous
    // run of them.
    std::vector<std::uint32_t> _faces;
};

template <typename Visitor>
void BVH::query(std::size_t u, std::size_t v, double u0, double v0, Visitor&& visit) const
{
    if (_nodes.empty()) return;

    // The tree is never deeper than maxDepth and only one sibling per level is
    // ever left waiting, so the stack cannot outgrow this.
    std::uint32_t stack[maxDepth + 2];
    int depth = 0;
    stack[depth++] = 0;

    while (depth > 0)
    {
        const Node& node = _nodes[stack[--depth]];

        // Node AABBs are float; the sample is double. Expand by one float ULP
        // so a sample that sits on a shared edge is not rejected here and then
        // accepted by the exact triangle test (or the reverse).
        const double minU = std::nextafter(node.min[u], -std::numeric_limits<float>::infinity());
        const double maxU = std::nextafter(node.max[u], std::numeric_limits<float>::infinity());
        const double minV = std::nextafter(node.min[v], -std::numeric_limits<float>::infinity());
        const double maxV = std::nextafter(node.max[v], std::numeric_limits<float>::infinity());
        if (u0 < minU || u0 > maxU || v0 < minV || v0 > maxV)
            continue;

        if (node.faceCount != 0)
        {
            for (std::uint32_t i = 0; i < node.faceCount; ++i)
                visit(_faces[node.firstChildOrFace + i]);
            continue;
        }

        stack[depth++] = node.firstChildOrFace;
        stack[depth++] = node.firstChildOrFace + 1;
    }
}

template <typename Visitor>
void BVH::queryRay(const double origin[3], const double invDir[3],
                   double tMin, double tMax, Visitor&& visit) const
{
    if (_nodes.empty()) return;

    std::uint32_t stack[maxDepth + 2];
    int depth = 0;
    stack[depth++] = 0;

    while (depth > 0)
    {
        const Node& node = _nodes[stack[--depth]];

        double lo = tMin;
        double hi = tMax;
        bool hit = true;
        for (int i = 0; i < 3; ++i)
        {
            double t0 = (static_cast<double>(node.min[i]) - origin[i]) * invDir[i];
            double t1 = (static_cast<double>(node.max[i]) - origin[i]) * invDir[i];
            if (invDir[i] < 0.0)
            {
                const double tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            if (t0 > lo) lo = t0;
            if (t1 < hi) hi = t1;
            if (hi < lo)
            {
                hit = false;
                break;
            }
        }
        if (!hit) continue;

        if (node.faceCount != 0)
        {
            for (std::uint32_t i = 0; i < node.faceCount; ++i)
                visit(_faces[node.firstChildOrFace + i]);
            continue;
        }

        stack[depth++] = node.firstChildOrFace;
        stack[depth++] = node.firstChildOrFace + 1;
    }
}

} // namespace app
