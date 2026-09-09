// TriangleMesh - the unwelded triangle soup every importer produces.
//
// Both the STL and 3MF readers hand back one of these, and BRep::fromTriangles
// turns it into topology. Vertices are per-triangle and not shared: welding
// them is the BRep's job, since that is what recovers the shared edges.
#pragma once

#include <string>
#include <vector>

#include <vsg/maths/vec3.h>

namespace app
{

// One facet: a face normal and three corner positions. The normal is what the
// source file declared, or is derived from the winding when it did not; nothing
// downstream relies on it, because the BRep recomputes area-weighted normals.
struct MeshTriangle
{
    vsg::vec3 normal;
    vsg::vec3 v0;
    vsg::vec3 v1;
    vsg::vec3 v2;
};

struct TriangleMesh
{
    std::string name;
    std::vector<MeshTriangle> triangles;
};

} // namespace app
