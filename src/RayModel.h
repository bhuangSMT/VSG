// RayModel - the solid sampled by axis-aligned rays cast through a BRep.
//
// For each of the three axis directions a regular grid of rays is cast across
// the model's bounding box. Every cast collects its intersections with the
// surface and pairs them up (1st-2nd, 3rd-4th, ...) so that each resulting Ray
// spans a stretch of solid material. The spans from one cast form a RayChain.
#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <vector>

#include <vsg/maths/mat4.h>

#include "BooleanOp.h"
#include "BoundingBox.h"
#include "Ray.h"

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

    // Cast rays through brep. resolution holds the sampling step along x, y and
    // z; a cast along one axis is stepped by the resolution of the two other
    // axes, so the x resolution spaces the rays cast along y and z, and so on.
    //
    // Casts that miss the model produce no chain, so the stored chains are all
    // non-empty. Throws std::invalid_argument if a resolution is not positive,
    // or if it is finer than can be cast; pass it through clampResolution
    // first to avoid the latter.
    static RayModel fromBRep(const BRep& brep, const Point3d& resolution);

    // The finest resolution that will still cast through brep, at or coarser
    // than requested, keeping the ratio between the axes. Returns requested
    // unchanged when it already fits.
    //
    // This is only ever a suggestion: nothing coarsens a resolution behind the
    // caller's back. Being too fine to *draw* is not a reason to coarsen at all
    // — the renderer thins a model out by grid layers and copes with any
    // density — so the only ceiling is what the cast can hold in memory, and
    // this is what to tell someone who has asked to go past it.
    static Point3d finestCastableResolution(const BRep& brep, const Point3d& requested);

    // Combine this model with a SweptVolume (see RayBoolean). Returns a new
    // model; this one is left unchanged so the original cast can be reused.
    RayModel withBoolean(const SweptVolume& sweep,
                         BooleanOp op,
                         const vsg::dmat4& modelToWorld) const;

    // axis is 0 for x, 1 for y, 2 for z.
    const std::vector<RayChain>& chains(std::size_t axis) const { return _chains[axis]; }

    const std::vector<RayChain>& xChains() const { return _chains[0]; }
    const std::vector<RayChain>& yChains() const { return _chains[1]; }
    const std::vector<RayChain>& zChains() const { return _chains[2]; }

    const BoundingBox& bounds() const { return _bounds; }
    const Point3d& resolution() const { return _resolution; }

    std::size_t chainCount() const;
    std::size_t rayCount() const;

    // Thinning the model out for display.
    //
    // A fine cast is worth keeping as geometry but is far more than a renderer
    // can hold: at a tenth of a millimetre a small part runs to millions of
    // rays, and one splat quad per endpoint would be gigabytes of vertices.
    // Dropping whole grid layers -- keeping every second line, then every
    // fourth -- leaves exactly the casts a coarser resolution would have made,
    // at their original precision, so the picture is the same one a coarser
    // cast would have drawn while the model itself stays fine.

    // How far apart the kept layers may get before the search gives up.
    static constexpr int maxStride = 1024;

    // The smallest stride whose kept rays number no more than maxRays, so the
    // drawn density lands near the budget whatever the cast resolution was.
    // Returns 1 when the model already fits.
    int strideForRayBudget(std::size_t maxRays) const;

    // How many rays survive at the given stride, for sizing buffers.
    std::size_t rayCountAtStride(int stride) const;

    // Drop spans shorter than the sampling step along their cast axis. Empty
    // chains are removed afterwards. Locks the chain mutex while rewriting.
    void removeDegenerateRays();

    // Hold while reading or writing _chains from another thread.
    std::unique_lock<std::recursive_mutex> lockChains() const;

private:
    friend RayModel applyBoolean(const RayModel& source,
                                 const SweptVolume& sweep,
                                 BooleanOp op,
                                 const vsg::dmat4& modelToWorld);

    // Indexed by direction: 0 = rays parallel to x, 1 = to y, 2 = to z.
    std::array<std::vector<RayChain>, 3> _chains;

    BoundingBox _bounds;
    Point3d _resolution{0.0, 0.0, 0.0};

    // Guards chain mutation (cleanup) against concurrent readers (draw / boolean).
    // Recursive so rayCount() can be called while a draw path already holds the lock.
    mutable std::recursive_mutex _chainMutex;
};

} // namespace app
