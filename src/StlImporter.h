// StlImporter - reads ASCII and binary STL files into a simple triangle soup.
#pragma once

#include <string>

#include "TriangleMesh.h"

namespace app
{

// Reads STL files (auto-detecting ASCII vs binary).
// Throws std::runtime_error on any I/O or parse failure.
class StlImporter
{
public:
    TriangleMesh import(const std::string& filename) const;

private:
    TriangleMesh importBinary(const std::string& filename) const;
    TriangleMesh importAscii(const std::string& filename) const;

    // Returns true if the file layout matches the binary STL size formula.
    static bool looksLikeBinary(const std::string& filename);
};

} // namespace app
