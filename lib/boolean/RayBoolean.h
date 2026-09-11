// RayBoolean - combine a RayModel with a SweptVolume along the cast rays.
//
// For each present RayGrid, the sweep AABB maps to a dense (iu, iv) window.
// Only those slots run BVH triangle tests; hits are quantized to int32 ticks
// and subtracted/unioned into that grid's IntervalPool.
//
// Union that grows the stock AABB appends new lattice rows/columns so the
// corner region is sampled, rather than only stretching existing X/Y spans.
#pragma once

#include <cstdint>

#include <vsg/maths/mat4.h>

#include "BooleanOp.h"
#include "BoundingBox.h"
#include "RayGrid.h"
#include "RayModel.h"
#include "SweptVolume.h"

namespace app
{

RayModel applyBoolean(const RayModel& source,
                      const SweptVolume& sweep,
                      BooleanOp op,
                      const vsg::dmat4& modelToWorld);

// Mutate model grids in place (dirty AABB window only). No deep copy.
void applyBooleanInPlace(RayModel& model,
                         const SweptVolume& sweep,
                         BooleanOp op,
                         const vsg::dmat4& modelToWorld);

// Map a world-space AABB through worldToModel (8 corners) into model space.
BoundingBox modelAabbFromWorld(const BoundingBox& worldBounds,
                               const vsg::dmat4& worldToModel);

// Inclusive lateral index window for one RayGrid covering modelAabb.
// Returns false when the grid or box is empty.
bool gridWindowFromModelAabb(const RayGrid& grid,
                             const BoundingBox& modelAabb,
                             std::uint32_t& iu0, std::uint32_t& iu1,
                             std::uint32_t& iv0, std::uint32_t& iv1);

bool gridWindowFromWorldAabb(const RayGrid& grid,
                             const BoundingBox& worldBounds,
                             const vsg::dmat4& worldToModel,
                             std::uint32_t& iu0, std::uint32_t& iu1,
                             std::uint32_t& iv0, std::uint32_t& iv1);

} // namespace app
