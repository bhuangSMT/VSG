// ThreeMfImporter - reads the geometry out of a 3MF package.
//
// A 3MF file is an OPC package: a ZIP archive whose 3D/3dmodel.model part is an
// XML description of the model. Only geometry is read. Materials, colours,
// textures and metadata are all ignored, because everything downstream works
// from topology alone.
//
// The build section is flattened: every build item is instantiated with its
// transform, components are followed recursively, and the result is one
// triangle soup in millimetres regardless of the unit the file declared.
#pragma once

#include <string>

#include "TriangleMesh.h"

namespace app
{

class ThreeMfImporter
{
public:
    // Throws std::runtime_error on any I/O, ZIP or XML failure, and on the
    // parts of the format that are not supported (ZIP64 and encrypted
    // archives), rather than returning a partial mesh.
    TriangleMesh import(const std::string& filename) const;
};

} // namespace app
