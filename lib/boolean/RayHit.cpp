#include "RayHit.h"

#include <algorithm>
#include <cmath>

namespace app
{

void clusterMergeHits(std::vector<RayHit>& hits, std::size_t axis, double tolerance)
{
    if (hits.size() < 2) return;

    std::sort(hits.begin(), hits.end(),
              [](const RayHit& lhs, const RayHit& rhs) { return lhs.along < rhs.along; });

    std::vector<RayHit> merged;
    merged.reserve(hits.size());

    auto cluster = hits.begin();
    while (cluster != hits.end())
    {
        auto clusterEnd = cluster + 1;
        while (clusterEnd != hits.end() &&
               (clusterEnd->along - (clusterEnd - 1)->along) <= tolerance)
            ++clusterEnd;

        RayHit best = *cluster;
        float bestScore = std::abs(best.normal[axis]);
        for (auto hit = cluster + 1; hit != clusterEnd; ++hit)
        {
            const float score = std::abs(hit->normal[axis]);
            if (score > bestScore)
            {
                best = *hit;
                bestScore = score;
            }
        }
        merged.push_back(best);

        cluster = clusterEnd;
    }

    hits.swap(merged);
}

void intervalsFromHits(std::vector<RayHit>& hits,
                       const RayGrid& grid,
                       std::size_t axis,
                       double mergeTol,
                       bool fromBoolean,
                       std::vector<Interval>& out,
                       PairingStats* stats)
{
    clusterMergeHits(hits, axis, mergeTol);

    if (stats && (hits.size() % 2 != 0))
        ++stats->oddHitRays;

    out.clear();
    for (auto h = decltype(hits.size()){0}; h + 1 < hits.size(); h += 2)
    {
        Interval iv;
        iv.begin = grid.toTick(hits[h].along);
        iv.end = grid.toTick(hits[h + 1].along);
        iv.beginNormal = hits[h].normal;
        iv.endNormal = hits[h + 1].normal;
        if (hits[h].motionCap) iv.setCapBegin(true);
        if (hits[h + 1].motionCap) iv.setCapEnd(true);
        if (fromBoolean)
        {
            iv.setFromBoolean(true);
            iv.setCutBegin(true);
            iv.setCutEnd(true);
        }

        // 1-tick spans are pairing noise: two almost-coincident endpoints
        // draw as orphan Gaussians. Keep only spans that cover 2+ ticks.
        if (iv.end - iv.begin >= 2)
            out.push_back(iv);
        else if (stats)
            ++stats->sliversDropped;
    }

    if (stats && (hits.size() % 2 != 0))
        ++stats->unmatchedLeave;
}

} // namespace app
