// Ray primitives produced by casting through a BRep.
#pragma once

#include <array>
#include <vector>

namespace app
{

// A point in model space. Doubles, so that intersection coordinates keep their
// precision independently of the float vertex data they are derived from.
using Point3d = std::array<double, 3>;

// A unit direction. Floats: a normal only ever feeds shading, which cannot see
// the difference, and there are two of them per Ray.
using Normal3f = std::array<float, 3>;

// One span of a cast ray that lies inside the model: it begins where the ray
// enters the surface and ends where it next leaves.
//
// The normals are the outward surface normals of the faces met at each end,
// oriented against the direction of travel at the start and along it at the
// end. They are carried here rather than recovered later because the cast is
// the only point at which the face that was hit is known.
struct Ray
{
    Point3d startPoint{0.0, 0.0, 0.0};
    Point3d endPoint{0.0, 0.0, 0.0};
    Normal3f startNormal{0.0f, 0.0f, 0.0f};
    Normal3f endNormal{0.0f, 0.0f, 0.0f};

    // Set on spans rewritten by a boolean with the swept volume; the renderer
    // draws those in the tool colour.
    bool fromBoolean = false;
};

// Every span produced by a single cast: the ray enters the solid at the first
// intersection and leaves at the second (forming the first Ray), re-enters at
// the third and leaves at the fourth, and so on.
//
// The chain also records where on the sampling grid its cast sat. Those indices
// are what let a renderer thin the model out by dropping whole grid layers,
// keeping every second or fourth line in each direction, which is the same set
// of casts a coarser resolution would have produced.
struct RayChain
{
    int u = 0;
    int v = 0;
    std::vector<Ray> rays;
};

} // namespace app
