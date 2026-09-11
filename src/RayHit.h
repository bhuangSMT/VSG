// RayHit - merge and pair axis-aligned triangle hits into solid intervals.
//
// Shared by the BRep caster and the swept-volume boolean. Duplicate edge hits
// become one sample; pairing is even-odd after that merge (STL winding is not
// trusted). One-tick spans are dropped so they never become orphan Gaussians.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Ray.h"
#include "RayGrid.h"

namespace app
{

struct RayHit
{
    double along = 0.0;
    Normal3f normal{0.0f, 0.0f, 0.0f};
};

struct PairingStats
{
    long long oddHitRays = 0;
    long long unmatchedEnter = 0;
    long long unmatchedLeave = 0;
    long long sliversDropped = 0;

    PairingStats& operator+=(const PairingStats& other)
    {
        oddHitRays += other.oddHitRays;
        unmatchedEnter += other.unmatchedEnter;
        unmatchedLeave += other.unmatchedLeave;
        sliversDropped += other.sliversDropped;
        return *this;
    }
};

// Inclusive barycentric test with a small epsilon so a sample on a shared
// edge is not rejected by both triangles.
inline bool barycentricInside(double w0, double w1, double w2)
{
    constexpr double eps = 1.0e-8;
    return w0 >= -eps && w1 >= -eps && w2 >= -eps;
}

inline bool pointInUvBounds(double u0, double v0,
                            double loU, double hiU, double loV, double hiV)
{
    constexpr double eps = 1.0e-8;
    return u0 >= loU - eps && u0 <= hiU + eps && v0 >= loV - eps && v0 <= hiV + eps;
}

// Collapse hits whose along-gap is within tolerance. Each cluster keeps one
// sample (strongest |normal[axis]|). Coincident opposite faces must not become
// two hits: that shifts even-odd parity and spans empty space between members.
void clusterMergeHits(std::vector<RayHit>& hits, std::size_t axis, double tolerance);

// Cluster-merge, then pair in along order (even-odd). STL soups are not
// reliably oriented, so winding is not used. Intervals shorter than 2 ticks
// are dropped. fromBoolean tags cutter spans.
void intervalsFromHits(std::vector<RayHit>& hits,
                       const RayGrid& grid,
                       std::size_t axis,
                       double mergeTol,
                       bool fromBoolean,
                       std::vector<Interval>& out,
                       PairingStats* stats = nullptr);

} // namespace app
