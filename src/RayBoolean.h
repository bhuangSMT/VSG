// RayBoolean - combine a RayModel with a SweptVolume along the cast rays.
//
// Uses the SweptVolume BVH to skip RayChains that cannot meet the sweep, then
// splits each overlapping Ray by the solid intervals of the sweep along that
// cast. Subtraction removes those intervals; union merges them in.
#pragma once

#include <vsg/maths/mat4.h>

#include "BooleanOp.h"
#include "RayModel.h"
#include "SweptVolume.h"

namespace app
{

// Returns a new RayModel derived from source. modelToWorld maps RayModel
// coordinates into the space the SweptVolume (and its BVH) live in. Chains that
// miss the sweep are copied unchanged. Op::None returns a copy of source.
RayModel applyBoolean(const RayModel& source,
                      const SweptVolume& sweep,
                      BooleanOp op,
                      const vsg::dmat4& modelToWorld);

} // namespace app
