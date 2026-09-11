// BRep - a boundary representation of a triangle mesh.
//
// Topology is stored in CSR (Compressed Sparse Row) form as a face->vertex
// incidence matrix:
//
//   faceOffsets  : row pointers, size = faceCount + 1
//   faceVertices : column indices (vertex ids), size = faceOffsets[faceCount]
//
// The vertex ids for face f are:
//   faceVertices[ faceOffsets[f] .. faceOffsets[f+1] )
//
// This generalises to arbitrary polygons; for STL every row has length 3.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vsg/maths/vec3.h>

#include "BVH.h"
#include "TriangleMesh.h"

namespace app
{

class BRep
{
public:
    struct ValidationResult
    {
        bool watertight = false;
        std::size_t boundaryEdgeCount = 0;    // edges incident to exactly 1 face
        std::size_t nonManifoldEdgeCount = 0; // edges incident to more than 2 faces
        std::size_t openEdgeCount = 0;        // total problematic edges
        std::size_t degenerateFaceCount = 0;  // faces skipped during construction
        std::size_t weldedVertexCount = 0;    // coincident corners merged
    };

    // Build a BRep from an STL mesh, welding coincident vertices so that
    // topology (shared edges) can be recovered. The face BVH is built here too,
    // so anything holding a BRep can cast against it without a further step.
    static BRep fromTriangles(const TriangleMesh& mesh);

    std::size_t vertexCount() const { return _vertices.size(); }
    std::size_t faceCount() const { return _faceOffsets.empty() ? 0 : _faceOffsets.size() - 1; }

    const std::vector<vsg::vec3>& vertices() const { return _vertices; }
    const std::vector<std::uint32_t>& faceOffsets() const { return _faceOffsets; }
    const std::vector<std::uint32_t>& faceVertices() const { return _faceVertices; }

    // Face hierarchy over the topology above, used to narrow down which
    // triangles a ray can possibly meet. Empty only when there are no faces.
    const BVH& bvh() const { return _bvh; }

    // Watertight / manifold check over the CSR topology. A mesh is considered
    // watertight when every edge is shared by exactly two faces (no boundary
    // edges and no non-manifold edges).
    ValidationResult validate() const;

    // The unique undirected edges of the CSR topology, flattened into vertex
    // index pairs (a0, b0, a1, b1, ...). An edge shared by two faces appears
    // once, so this is directly usable as a line-list index buffer.
    std::vector<std::uint32_t> extractEdgeIndices() const;

private:
    std::vector<vsg::vec3> _vertices;
    std::vector<std::uint32_t> _faceOffsets;
    std::vector<std::uint32_t> _faceVertices;
    BVH _bvh;
    std::size_t _degenerateFaceCount = 0;
    std::size_t _weldedVertexCount = 0;
};

} // namespace app
