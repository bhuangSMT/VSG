// RayModel - persistent stock sampled on independent X/Y/Z RayGrids.
//
// Each grid is sized from the BRep AABB + resolution, stores remaining solid as
// int32 tick intervals in its own IntervalPool, and can be omitted later without
// affecting the other axes.
#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <optional>

#include <vsg/maths/mat4.h>

#include "BooleanOp.h"
#include "BoundingBox.h"
#include "Ray.h"
#include "RayGrid.h"
#include "RayHit.h"

namespace app
{

class BRep;
class SweptVolume;

class RayModel
{
public:
    RayModel() = default;
    RayModel(const RayModel&) = delete;
    RayModel& operator=(const RayModel&) = delete;
    RayModel(RayModel&& other) noexcept;
    RayModel& operator=(RayModel&& other) noexcept;

    static RayModel fromBRep(const BRep& brep, const Point3d& resolution);
    static Point3d finestCastableResolution(const BRep& brep, const Point3d& requested);

    // Deep-copy grids into a new model (one-time fork from source stock).
    RayModel clone() const;

    // Copy-then-mutate (allocates a new model). Prefer booleanInPlace for cuts.
    RayModel withBoolean(const SweptVolume& sweep,
                         BooleanOp op,
                         const vsg::dmat4& modelToWorld) const;

    // Mutate this model's grids in place (no deep copy).
    void booleanInPlace(const SweptVolume& sweep,
                        BooleanOp op,
                        const vsg::dmat4& modelToWorld);

    // Hollow every solid interval: keep an inward wall of `thickness` (model
    // units) at each endpoint. Degenerate walls and overlapping wall pairs from
    // the same parent are dropped. Throws if thickness is not positive.
    void shellInPlace(double thickness);

    // nullptr when that axis was not built.
    const RayGrid* grid(std::size_t axis) const;
    RayGrid* grid(std::size_t axis);

    const BoundingBox& bounds() const { return _bounds; }
    const Point3d& resolution() const { return _resolution; }

    std::size_t rayCount() const; // total remaining intervals across grids
    std::size_t intervalCount() const { return rayCount(); }

    // Hit pairing leftovers from the last fromBRep / booleanInPlace.
    const PairingStats& pairingStats() const { return _pairingStats; }

    // Cells visited by the last booleanInPlace (per-triangle UV footprints).
    std::size_t lastDirtyCellCount() const { return _lastDirtyCellCount; }

    static constexpr int maxStride = 1024;
    int strideForRayBudget(std::size_t maxRays) const;
    std::size_t rayCountAtStride(int stride) const;

    // Interval count on stride-aligned cells whose lateral sample falls in aabb.
    // Falls back to rayCountAtStride when aabb is empty/invalid.
    std::size_t rayCountInAabbAtStride(const BoundingBox& aabb, int stride) const;
    // Smallest stride whose visible (aabb) interval count fits maxRays.
    int strideForVisibleBudget(const BoundingBox& aabb, std::size_t maxRays) const;

    std::unique_lock<std::recursive_mutex> lockChains() const;

private:
    friend RayModel applyBoolean(const RayModel& source,
                                 const SweptVolume& sweep,
                                 BooleanOp op,
                                 const vsg::dmat4& modelToWorld);
    friend void applyBooleanInPlace(RayModel& model,
                                    const SweptVolume& sweep,
                                    BooleanOp op,
                                    const vsg::dmat4& modelToWorld);

    std::array<std::optional<RayGrid>, 3> _grids;
    BoundingBox _bounds;
    Point3d _resolution{0.0, 0.0, 0.0};
    PairingStats _pairingStats;
    std::size_t _lastDirtyCellCount = 0;

    mutable std::recursive_mutex _chainMutex;
};

} // namespace app
