// ModelPick - cast a ray through the BRep and report the nearest hit.
//
// Used to place the mouse-driven tool. The BVH's existing axis-aligned query is
// not enough here: the mouse ray can come from any direction, so this walks the
// same tree with a general ray/AABB test and finishes with Möller–Trumbore on
// the leaf triangles.
#pragma once

#include <optional>

#include <vsg/maths/vec3.h>

#include "BRep.h"

namespace app
{

struct ModelHit
{
    vsg::dvec3 position{0.0, 0.0, 0.0};
    vsg::dvec3 normal{0.0, 0.0, 1.0};
    double t = 0.0;
};

// Closest intersection of the ray origin + t * direction with the BRep, for
// t in (tMin, tMax). direction need not be unit length; t is in the same units
// as the direction vector's length. The reported normal points to the side the
// ray approached from (against the ray), so it is suitable as an outward tool
// axis when the ray is coming from the camera.
std::optional<ModelHit> pickBRep(const BRep& brep,
                                 const vsg::dvec3& origin,
                                 const vsg::dvec3& direction,
                                 double tMin = 0.0,
                                 double tMax = 1.0e9);

// Intersection of the ray with the plane through planePoint with the given
// normal. Empty when the ray is parallel to the plane or the hit falls outside
// [tMin, tMax]. The reported normal faces the side the ray approached from.
std::optional<ModelHit> pickPlane(const vsg::dvec3& origin,
                                  const vsg::dvec3& direction,
                                  const vsg::dvec3& planePoint,
                                  const vsg::dvec3& planeNormal,
                                  double tMin = 0.0,
                                  double tMax = 1.0e9);

} // namespace app
