// GaussianSplatCache - sparse GPU splat buffers for Ray-GS.
#include "GaussianSplatCache.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <tbb/parallel_for.h>

#include "RayBoolean.h"
#include "RayGrid.h"

namespace app
{
namespace
{

vsg::vec3 normalOrAxis(const Normal3f& n, std::size_t axis, bool enter)
{
    const float len2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (len2 < 1.0e-12f)
    {
        vsg::vec3 fallback(0.0f, 0.0f, 0.0f);
        fallback[static_cast<int>(axis)] = enter ? -1.0f : 1.0f;
        return fallback;
    }
    return vsg::vec3(n[0], n[1], n[2]);
}

bool normalValid(const Normal3f& n)
{
    return n[0] * n[0] + n[1] * n[1] + n[2] * n[2] >= 1.0e-12f;
}

// Closest solid endpoint on the enter (begin) or exit (end) side of a ray cell,
// plus the normal the cast recorded there (degenerate when it recorded none).
bool sampleFaceEndpoint(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                        std::uint32_t iv, bool enter, double alongTarget, double maxAlongDelta,
                        Point3d& out, Normal3f* outNormal = nullptr)
{
    if (iu >= grid.width || iv >= grid.height) return false;
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return false;

    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid.sampleU(iu);
    const double v0 = grid.sampleV(iv);

    double best = maxAlongDelta;
    bool found = false;
    for (const Interval& span : grid.pool.span(slot))
    {
        if (!span.hasSolidLength()) continue;
        const double along = enter ? grid.fromTick(span.begin) : grid.fromTick(span.end);
        const double d = std::abs(along - alongTarget);
        if (d > best) continue;
        best = d;
        out = Point3d{0.0, 0.0, 0.0};
        out[axis] = along;
        out[u] = u0;
        out[v] = v0;
        if (outNormal) *outNormal = enter ? span.beginNormal : span.endNormal;
        found = true;
    }
    return found;
}

// Nearest endpoint in a cell that carries a usable recorded normal. Separate
// from sampleFaceEndpoint because inheriting a direction tolerates a far looser
// distance than fitting a plane does, and because the nearest endpoint of all
// may be a sliver with no normal while one just behind it has a good one.
bool sampleRecordedNormal(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                          std::uint32_t iv, bool enter, double alongTarget,
                          double maxAlongDelta, Normal3f& out)
{
    if (iu >= grid.width || iv >= grid.height) return false;
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return false;

    double best = maxAlongDelta;
    bool found = false;
    for (const Interval& span : grid.pool.span(slot))
    {
        const Normal3f& rec = enter ? span.beginNormal : span.endNormal;
        if (!normalValid(rec)) continue;
        const double at = enter ? grid.fromTick(span.begin) : grid.fromTick(span.end);
        const double d = std::abs(at - alongTarget);
        if (d > best) continue;
        best = d;
        out = rec;
        found = true;
    }
    return found;
}

vsg::vec3 estimateNormalFromNeighbors(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                                      std::uint32_t iv, int stride, bool enter, double along)
{
    const int s = stride < 1 ? 1 : stride;
    const double maxDelta = 3.0 * static_cast<double>(s) *
                            std::max(grid.spacingU, grid.spacingV);
    Point3d left, right, up, down;
    Normal3f recL{}, recR{}, recU{}, recD{};
    const bool hasL =
        iu >= static_cast<std::uint32_t>(s) &&
        sampleFaceEndpoint(grid, axis, iu - static_cast<std::uint32_t>(s), iv, enter, along,
                           maxDelta, left, &recL);
    const bool hasR =
        iu + static_cast<std::uint32_t>(s) < grid.width &&
        sampleFaceEndpoint(grid, axis, iu + static_cast<std::uint32_t>(s), iv, enter, along,
                           maxDelta, right, &recR);
    const bool hasU =
        iv >= static_cast<std::uint32_t>(s) &&
        sampleFaceEndpoint(grid, axis, iu, iv - static_cast<std::uint32_t>(s), enter, along,
                           maxDelta, up, &recU);
    const bool hasD =
        iv + static_cast<std::uint32_t>(s) < grid.height &&
        sampleFaceEndpoint(grid, axis, iu, iv + static_cast<std::uint32_t>(s), enter, along,
                           maxDelta, down, &recD);

    vsg::vec3 axisDir(0.0f, 0.0f, 0.0f);
    axisDir[static_cast<int>(axis)] = enter ? -1.0f : 1.0f;

    // This endpoint, from the cell's own sample position. Lets a one-sided
    // difference stand in where a neighbour is missing, which is the normal
    // case along a surface edge: losing a neighbour should cost accuracy, not
    // the whole normal. Snapping to axisDir instead leaves the disc with a
    // normal lying flat in the surface, which shades as a dark speck.
    Point3d centre{0.0, 0.0, 0.0};
    centre[axis] = along;
    centre[(axis + 1) % 3] = grid.sampleU(iu);
    centre[(axis + 2) % 3] = grid.sampleV(iv);

    auto delta = [](const Point3d& a, const Point3d& b) {
        return vsg::vec3(static_cast<float>(a[0] - b[0]), static_cast<float>(a[1] - b[1]),
                         static_cast<float>(a[2] - b[2]));
    };

    vsg::vec3 dx, dy;
    bool haveDx = true;
    bool haveDy = true;
    if (hasL && hasR) dx = delta(right, left);
    else if (hasR) dx = delta(right, centre);
    else if (hasL) dx = delta(centre, left);
    else haveDx = false;

    if (hasU && hasD) dy = delta(down, up);
    else if (hasD) dy = delta(down, centre);
    else if (hasU) dy = delta(centre, up);
    else haveDy = false;

    if (haveDx && haveDy)
    {
        vsg::vec3 n = vsg::cross(dx, dy);
        const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 >= 1.0e-12f)
        {
            n *= 1.0f / std::sqrt(len2);
            // Match axis-fallback orientation (enter → -axis, exit → +axis).
            if (vsg::dot(n, axisDir) < 0.0f) n = -n;
            return n;
        }
    }

    // No plane to fit: inherit the neighbours' recorded normals. Only recorded
    // ones — an estimate would recurse back into here through endpointNormal.
    // Signs are aligned to the first contributor rather than to axisDir, which
    // coin-flips on a face the rays grazed.
    //
    // The window widens here: maxDelta is sized to keep a plane fit honest, but
    // on a face the rays graze the neighbouring endpoint sits far along the ray,
    // so that window rejects every neighbour and leaves nothing to inherit.
    // A direction stays useful over a much longer reach than a position does.
    const double inheritDelta = 4.0 * maxDelta;
    const auto su = static_cast<std::uint32_t>(s);
    vsg::vec3 accum(0.0f, 0.0f, 0.0f);
    float accumLen2 = 0.0f;
    auto inherit = [&](bool inRange, std::uint32_t nu, std::uint32_t nv,
                       bool have, const Normal3f& rec) {
        Normal3f use = rec;
        if (!have || !normalValid(use))
        {
            if (!inRange) return;
            if (!sampleRecordedNormal(grid, axis, nu, nv, enter, along, inheritDelta, use))
                return;
        }
        vsg::vec3 n(use[0], use[1], use[2]);
        if (accumLen2 > 0.0f && vsg::dot(n, accum) < 0.0f) n = -n;
        accum += n;
        accumLen2 = accum.x * accum.x + accum.y * accum.y + accum.z * accum.z;
    };
    inherit(iu >= su, iu >= su ? iu - su : 0u, iv, hasL, recL);
    inherit(iu + su < grid.width, iu + su, iv, hasR, recR);
    inherit(iv >= su, iu, iv >= su ? iv - su : 0u, hasU, recU);
    inherit(iv + su < grid.height, iu, iv + su, hasD, recD);
    if (accumLen2 >= 1.0e-12f) return accum * (1.0f / std::sqrt(accumLen2));

    // Last resort before the axis: this cell's own other endpoints. A sliver
    // with no normal often shares a ray with a span that has one.
    Normal3f own{};
    if (sampleRecordedNormal(grid, axis, iu, iv, enter, along, inheritDelta, own) ||
        sampleRecordedNormal(grid, axis, iu, iv, !enter, along, inheritDelta, own))
    {
        vsg::vec3 n(own[0], own[1], own[2]);
        const float len2 = n.x * n.x + n.y * n.y + n.z * n.z;
        if (len2 >= 1.0e-12f) return n * (1.0f / std::sqrt(len2));
    }

    // Genuinely isolated endpoint: nothing better than the ray axis.
    return normalOrAxis(Normal3f{}, axis, enter);
}

vsg::vec3 endpointNormal(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                         std::uint32_t iv, int stride, const Normal3f& recorded, bool enter,
                         double along)
{
    if (normalValid(recorded)) return normalOrAxis(recorded, axis, enter);
    return estimateNormalFromNeighbors(grid, axis, iu, iv, stride, enter, along);
}

// Sample the normal of the nearest solid endpoint on the same face side in a
// neighbour cell. Returns false when the neighbour is empty / out of range.
bool sampleNeighborNormal(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                          std::uint32_t iv, int stride, bool enter, double along,
                          vsg::vec3& outNormal)
{
    if (iu >= grid.width || iv >= grid.height) return false;
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return false;

    const int s = stride < 1 ? 1 : stride;
    const double maxDelta = 3.0 * static_cast<double>(s) *
                            std::max(grid.spacingU, grid.spacingV);

    double best = maxDelta;
    bool found = false;
    Normal3f recorded{};
    double bestAlong = along;
    for (const Interval& span : grid.pool.span(slot))
    {
        if (!span.hasSolidLength()) continue;
        const double a = enter ? grid.fromTick(span.begin) : grid.fromTick(span.end);
        const double d = std::abs(a - along);
        if (d > best) continue;
        best = d;
        recorded = enter ? span.beginNormal : span.endNormal;
        bestAlong = a;
        found = true;
    }
    if (!found) return false;
    outNormal = endpointNormal(grid, axis, iu, iv, stride, recorded, enter, bestAlong);
    return true;
}

// bit0=+X bit1=-X bit2=+Y bit3=-Y bit4=+Z bit5=-Z
// Strength ramp: 0 while the neighbour normal is still near-parallel, 1 once
// the crease is unmistakable. A direction bit only fires past kEdgeBitMin so
// that gentle curvature leaves the disc perfectly round.
constexpr float kSmoothDot = 0.95f;
constexpr float kSharpDot = 0.20f;
constexpr float kEdgeBitMin = 0.10f;

float edgeStrengthFromDot(float d)
{
    return std::clamp((kSmoothDot - d) / (kSmoothDot - kSharpDot), 0.0f, 1.0f);
}

void setWorldAxisBit(std::uint8_t& mask, std::size_t worldAxis, bool positive)
{
    const unsigned bit = static_cast<unsigned>(worldAxis) * 2u + (positive ? 0u : 1u);
    mask = static_cast<std::uint8_t>(mask | (1u << bit));
}

struct EdgeInfo
{
    std::uint8_t mask = 0;
    float strength = 0.0f;
};

EdgeInfo computeEdgeMask(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                         std::uint32_t iv, int stride, bool enter, double along,
                         const vsg::vec3& normal)
{
    const int s = stride < 1 ? 1 : stride;
    const auto su = static_cast<std::uint32_t>(s);
    const std::size_t uAxis = (axis + 1) % 3;
    const std::size_t vAxis = (axis + 2) % 3;

    EdgeInfo info;
    auto consider = [&](bool have, const vsg::vec3& neighborN, std::size_t worldAxis,
                        bool positiveDir) {
        if (!have) return;
        const float strength = edgeStrengthFromDot(vsg::dot(normal, neighborN));
        info.strength = std::max(info.strength, strength);
        if (strength > kEdgeBitMin) setWorldAxisBit(info.mask, worldAxis, positiveDir);
    };

    vsg::vec3 nL, nR, nU, nD;
    const bool hasL =
        iu >= su && sampleNeighborNormal(grid, axis, iu - su, iv, stride, enter, along, nL);
    const bool hasR = iu + su < grid.width &&
                      sampleNeighborNormal(grid, axis, iu + su, iv, stride, enter, along, nR);
    const bool hasU =
        iv >= su && sampleNeighborNormal(grid, axis, iu, iv - su, stride, enter, along, nU);
    const bool hasD = iv + su < grid.height &&
                      sampleNeighborNormal(grid, axis, iu, iv + su, stride, enter, along, nD);

    // UV steps map to world ±uAxis / ±vAxis.
    consider(hasL, nL, uAxis, false);
    consider(hasR, nR, uAxis, true);
    consider(hasU, nU, vAxis, false);
    consider(hasD, nD, vAxis, true);

    // Along-axis: compare with the opposite face endpoint on the same ray when
    // it lies close enough to count as a sharp feature (thin wall / cut).
    const RaySlot& slot = grid.at(iu, iv);
    if (!slot.empty())
    {
        const double maxDelta = 3.0 * static_cast<double>(s) *
                                std::max(grid.spacingU, grid.spacingV);
        for (const Interval& span : grid.pool.span(slot))
        {
            if (!span.hasSolidLength()) continue;
            const double a = enter ? grid.fromTick(span.begin) : grid.fromTick(span.end);
            if (std::abs(a - along) > 1.0e-9) continue;
            const double other = enter ? grid.fromTick(span.end) : grid.fromTick(span.begin);
            if (std::abs(other - along) > maxDelta) continue;
            const Normal3f& rec = enter ? span.endNormal : span.beginNormal;
            const vsg::vec3 otherN =
                endpointNormal(grid, axis, iu, iv, stride, rec, !enter, other);
            // Discontinuity toward the other end along the ray axis.
            consider(true, otherN, axis, enter);
            break;
        }
    }
    return info;
}

std::uint32_t sampledCount(std::uint32_t extent, int stride)
{
    if (extent == 0 || stride <= 0) return 0;
    return (extent + static_cast<std::uint32_t>(stride) - 1) /
           static_cast<std::uint32_t>(stride);
}

std::uint32_t sampledIndex(std::uint32_t i, int stride)
{
    return i / static_cast<std::uint32_t>(stride);
}

// Interval length is in model space. The fitted splat radius is not — it has
// been divided by applyFit(). A leftover shorter than two display-cell
// diagonals puts its untagged stock end on the cut overlay.
double leftoverHideLength(const RayGrid& grid, int stride)
{
    const int s = (stride > 0) ? stride : 1;
    const double cellDiag = std::sqrt(static_cast<double>(grid.spacingU) *
                                          static_cast<double>(grid.spacingU) +
                                      static_cast<double>(grid.spacingV) *
                                          static_cast<double>(grid.spacingV));
    return 2.0 * cellDiag * static_cast<double>(s);
}

double bleedHideLength(const RayGrid& grid, int stride)
{
    const int s = (stride > 0) ? stride : 1;
    const double cellDiag = std::sqrt(static_cast<double>(grid.spacingU) *
                                          static_cast<double>(grid.spacingU) +
                                      static_cast<double>(grid.spacingV) *
                                          static_cast<double>(grid.spacingV));
    return cellDiag * static_cast<double>(s);
}

void skipSplatEnds(const Interval& span, double modelLength, double hideShorterThan,
                   double bleedHideLength, bool skipCutSplats, bool onCutRim,
                   bool& skipStart, bool& skipEnd)
{
    const bool cutStart = span.cutBegin();
    const bool cutEnd = span.cutEnd();
    // Interior cut ends are covered by the overlay mesh; rim cells keep dots
    // so the edge seals against stock Gaussians.
    skipStart = skipCutSplats && cutStart && !onCutRim;
    skipEnd = skipCutSplats && cutEnd && !onCutRim;
    // Short leftovers and bleed partners: still useful when cut dots are kept.
    if ((cutStart || cutEnd) && hideShorterThan > 0.0 && modelLength <= hideShorterThan)
    {
        skipStart = true;
        skipEnd = true;
    }
    // Untagged partner sitting on the overlay: hide it, keep a distant stock end.
    if (bleedHideLength > 0.0 && modelLength <= bleedHideLength)
    {
        if (cutStart && !cutEnd) skipEnd = true;
        if (cutEnd && !cutStart) skipStart = true;
    }
}

bool cellHasCutTag(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv)
{
    if (iu >= grid.width || iv >= grid.height) return false;
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return false;
    for (const Interval& span : grid.pool.span(slot))
    {
        if (!span.hasSolidLength()) continue;
        if (span.cutBegin() || span.cutEnd()) return true;
    }
    return false;
}

// Cut-region UV rim at packing stride: any missing/empty/non-cut neighbour.
bool cutCellOnRim(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv, int stride)
{
    if (stride < 1) stride = 1;
    const auto s = static_cast<std::uint32_t>(stride);
    auto neighborCut = [&](std::uint32_t nu, std::uint32_t nv) {
        return nu < grid.width && nv < grid.height && cellHasCutTag(grid, nu, nv);
    };
    if (iu < s || !neighborCut(iu - s, iv)) return true;
    if (iu + s >= grid.width || !neighborCut(iu + s, iv)) return true;
    if (iv < s || !neighborCut(iu, iv - s)) return true;
    if (iv + s >= grid.height || !neighborCut(iu, iv + s)) return true;
    return false;
}

bool onCoarseLattice(std::uint32_t iu, std::uint32_t iv, int coarseStride)
{
    if (coarseStride <= 1) return true;
    return static_cast<int>(iu) % coarseStride == 0 && static_cast<int>(iv) % coarseStride == 0;
}

double dist2ToEye(const Point3d& p, const vsg::dvec3& eye)
{
    const double dx = p[0] - eye.x;
    const double dy = p[1] - eye.y;
    const double dz = p[2] - eye.z;
    return dx * dx + dy * dy + dz * dz;
}

bool modelPointInView(const vsg::dmat4& modelToClip, double x, double y, double z, float margin)
{
    const vsg::dvec4 clip = modelToClip * vsg::dvec4(x, y, z, 1.0);
    if (clip.w <= 1.0e-12) return false;
    const double invW = 1.0 / clip.w;
    const double ndcX = clip.x * invW;
    const double ndcY = clip.y * invW;
    const double m = static_cast<double>(margin);
    return ndcX >= -1.0 - m && ndcX <= 1.0 + m && ndcY >= -1.0 - m && ndcY <= 1.0 + m;
}

// Keep far-face ends on the coarse lattice only (no densify, no full cull).
void applyViewEndPolicy(const Point3d& start, const Point3d& end, std::uint32_t iu,
                        std::uint32_t iv, const SplatViewCull& cull, bool& skipStart,
                        bool& skipEnd)
{
    if (!cull.enabled) return;
    const bool startNear = dist2ToEye(start, cull.eyeModel) <= dist2ToEye(end, cull.eyeModel);
    const bool coarse = onCoarseLattice(iu, iv, cull.coarseStride);
    if (startNear)
    {
        if (!coarse) skipEnd = true;
    }
    else
    {
        if (!coarse) skipStart = true;
    }
}

// Cell kept when either stock end along the cast is on screen.
bool cellInViewCull(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv,
                    const BoundingBox& stockBounds, const SplatViewCull& cull)
{
    if (!cull.enabled) return true;
    if (!stockBounds.valid()) return true;

    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    Point3d a{0.0, 0.0, 0.0};
    Point3d b{0.0, 0.0, 0.0};
    a[u] = b[u] = grid.sampleU(iu);
    a[v] = b[v] = grid.sampleV(iv);
    a[axis] = stockBounds.min()[axis];
    b[axis] = stockBounds.max()[axis];
    return modelPointInView(cull.modelToClip, a[0], a[1], a[2], cull.ndcMargin) ||
           modelPointInView(cull.modelToClip, b[0], b[1], b[2], cull.ndcMargin);
}

bool endpointInViewCull(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv, double along,
                        const SplatViewCull& cull)
{
    if (!cull.enabled) return true;
    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    Point3d sample{0.0, 0.0, 0.0};
    sample[u] = grid.sampleU(iu);
    sample[v] = grid.sampleV(iv);
    sample[axis] = along;
    return modelPointInView(cull.modelToClip, sample[0], sample[1], sample[2], cull.ndcMargin);
}

std::uint32_t endpointNeed(const RayGrid& grid, std::uint32_t iu, std::uint32_t iv,
                           int stride, bool skipCutSplats, const SplatViewCull& viewCull)
{
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return 0;

    const std::size_t axis = grid.axis;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid.sampleU(iu);
    const double v0 = grid.sampleV(iv);
    const double hide = leftoverHideLength(grid, stride);
    const bool onRim = !skipCutSplats || cutCellOnRim(grid, iu, iv, stride);
    std::uint32_t n = 0;
    for (const Interval& span : grid.pool.span(slot))
    {
        if (!span.hasSolidLength()) continue;
        Point3d start{0.0, 0.0, 0.0};
        Point3d end{0.0, 0.0, 0.0};
        start[axis] = grid.fromTick(span.begin);
        end[axis] = grid.fromTick(span.end);
        start[u] = end[u] = u0;
        start[v] = end[v] = v0;

        const double modelLength = end[axis] - start[axis];
        bool skipStart = false;
        bool skipEnd = false;
        skipSplatEnds(span, modelLength, hide, bleedHideLength(grid, stride), skipCutSplats,
                      onRim, skipStart, skipEnd);
        // Cut disks are the cut surface when mesh is off — never view-cull them.
        const bool protectStart = span.cutBegin() && !skipCutSplats;
        const bool protectEnd = span.cutEnd() && !skipCutSplats;
        if (!protectStart || !protectEnd)
        {
            bool vs = skipStart;
            bool ve = skipEnd;
            applyViewEndPolicy(start, end, iu, iv, viewCull, vs, ve);
            if (!protectStart) skipStart = vs;
            if (!protectEnd) skipEnd = ve;
        }
        if (!skipStart &&
            (protectStart || endpointInViewCull(grid, iu, iv, start[axis], viewCull)))
            ++n;
        if (!skipEnd &&
            (protectEnd || endpointInViewCull(grid, iu, iv, end[axis], viewCull)))
            ++n;
    }
    return n;
}

struct SectionPoint
{
    double along = 0.0;
    vsg::vec3 pos{0.0f, 0.0f, 0.0f};
    vsg::vec3 normal{0.0f, 0.0f, 1.0f};
};

struct SectionVert
{
    vsg::vec3 pos{0.0f, 0.0f, 0.0f};
    vsg::vec3 normal{0.0f, 0.0f, 1.0f};
};

void collectSectionPoints(const RayGrid& grid, std::size_t axis, std::uint32_t iu,
                          std::uint32_t iv, const BoundingBox& box, bool clipToAabb,
                          std::vector<SectionPoint>& out)
{
    out.clear();
    const RaySlot& slot = grid.at(iu, iv);
    if (slot.empty()) return;

    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid.sampleU(iu);
    const double v0 = grid.sampleV(iv);
    auto spans = grid.pool.span(slot);
    for (const Interval& span : spans)
    {
        if (!span.hasSolidLength()) continue;
        Point3d start{0.0, 0.0, 0.0};
        Point3d end{0.0, 0.0, 0.0};
        start[axis] = grid.fromTick(span.begin);
        end[axis] = grid.fromTick(span.end);
        start[u] = u0;
        end[u] = u0;
        start[v] = v0;
        end[v] = v0;
        // Inspection clips to the cutter AABB so leftover stock ends in that
        // box do not fill it as a square. The cut-face overlay walks a UV
        // window and takes every tagged leftover on those rays (steep faces
        // jump outside the 3D box between adjacent samples).
        if (span.cutBegin() && (!clipToAabb || !box.valid() || box.contains(start)))
        {
            out.push_back({start[axis],
                           vsg::vec3(static_cast<float>(start[0]),
                                     static_cast<float>(start[1]),
                                     static_cast<float>(start[2])),
                           normalOrAxis(span.beginNormal, axis, true)});
        }
        if (span.cutEnd() && (!clipToAabb || !box.valid() || box.contains(end)))
        {
            out.push_back({end[axis],
                           vsg::vec3(static_cast<float>(end[0]),
                                     static_cast<float>(end[1]),
                                     static_cast<float>(end[2])),
                           normalOrAxis(span.endNormal, axis, false)});
        }
    }
    std::sort(out.begin(), out.end(),
              [](const SectionPoint& a, const SectionPoint& b) { return a.along < b.along; });
}

void emitSectionQuad(const std::vector<SectionPoint>& here,
                     const std::vector<SectionPoint>& east,
                     const std::vector<SectionPoint>& north,
                     const std::vector<SectionPoint>& northEast,
                     float maxAlong,
                     float maxLink,
                     std::vector<std::array<SectionVert, 3>>& tris)
{
    if (here.empty() || maxAlong <= 0.0f || maxLink <= 0.0f) return;
    const float maxLink2 = maxLink * maxLink;

    auto dist2 = [](const vsg::vec3& a, const vsg::vec3& b) {
        const vsg::vec3 d = a - b;
        return d.x * d.x + d.y * d.y + d.z * d.z;
    };
    // Same cut sheet (top with top, bottom with bottom). along uses a looser
    // band; 3D links stay tight so hollow openings do not grow bridge quads.
    auto nearestOnSheet = [&](const std::vector<SectionPoint>& pts,
                              const SectionPoint& src) -> const SectionPoint* {
        const SectionPoint* best = nullptr;
        float bestD = maxLink2;
        for (const SectionPoint& p : pts)
        {
            if (std::abs(p.along - src.along) > static_cast<double>(maxAlong)) continue;
            const float d2 = dist2(p.pos, src.pos);
            if (d2 <= bestD)
            {
                bestD = d2;
                best = &p;
            }
        }
        return best;
    };

    auto emitOriented = [&](const vsg::vec3& a, const vsg::vec3& na,
                            vsg::vec3 b, vsg::vec3 nb,
                            vsg::vec3 c, vsg::vec3 nc,
                            const vsg::vec3& hint) {
        const vsg::vec3 fn = vsg::cross(b - a, c - a);
        if (vsg::dot(fn, hint) < 0.0f)
        {
            std::swap(b, c);
            std::swap(nb, nc);
        }
        tris.push_back({SectionVert{a, na}, SectionVert{b, nb}, SectionVert{c, nc}});
    };

    for (const SectionPoint& h : here)
    {
        const SectionPoint* e = nearestOnSheet(east, h);
        const SectionPoint* n = nearestOnSheet(north, h);
        if (!e || !n || dist2(e->pos, n->pos) > maxLink2) continue;

        const vsg::vec3 predict = e->pos + n->pos - h.pos;
        const SectionPoint* ne = nullptr;
        float bestNe = maxLink2;
        for (const SectionPoint& p : northEast)
        {
            if (std::abs(p.along - h.along) > static_cast<double>(maxAlong)) continue;
            if (dist2(p.pos, e->pos) > maxLink2 || dist2(p.pos, n->pos) > maxLink2) continue;
            const float d2 = dist2(p.pos, predict);
            if (d2 <= bestNe)
            {
                bestNe = d2;
                ne = &p;
            }
        }
        if (!ne) continue;

        // Opposite-hemisphere only: noisy rim / axis-fallback normals still
        // form quads; strongly flipped sheets do not.
        vsg::vec3 hint = h.normal + e->normal + n->normal + ne->normal;
        const float hintLen = vsg::length(hint);
        if (hintLen < 1.0e-12f) hint = h.normal;
        else hint /= hintLen;
        if (vsg::dot(h.normal, hint) < 0.0f || vsg::dot(e->normal, hint) < 0.0f ||
            vsg::dot(n->normal, hint) < 0.0f || vsg::dot(ne->normal, hint) < 0.0f)
            continue;

        emitOriented(h.pos, h.normal, e->pos, e->normal, n->pos, n->normal, hint);
        emitOriented(e->pos, e->normal, ne->pos, ne->normal, n->pos, n->normal, hint);
    }
}

struct SampledWindow
{
    bool valid = false;
    std::uint32_t su0 = 0;
    std::uint32_t su1 = 0;
    std::uint32_t sv0 = 0;
    std::uint32_t sv1 = 0;
};

struct CollectedTri
{
    std::array<SectionVert, 3> verts;
    std::uint8_t axis = 0;
    std::uint32_t su = 0;
    std::uint32_t sv = 0;
};

double vecComponent(const vsg::vec3& p, std::size_t axis)
{
    if (axis == 0) return static_cast<double>(p.x);
    if (axis == 1) return static_cast<double>(p.y);
    return static_cast<double>(p.z);
}

// Halo pads dirty UV windows so multi-row CL Re-run patches do not leave
// seams between steps. Link scale must cover adjacent cut samples at
// cutFaceStride without reopening hollow-bridge stitches (those are >> 3 cells).
constexpr int kCutFaceHaloCells = 6;
constexpr float kInspectionMaxEdgeScale = 2.5f;
constexpr float kCutFaceMaxEdgeScale = 3.5f;
constexpr float kCutFaceMaxLinkScale = 2.5f;

void growGridWindowByStride(const RayGrid& grid, int stride, int haloCells,
                            std::uint32_t& iu0, std::uint32_t& iu1,
                            std::uint32_t& iv0, std::uint32_t& iv1)
{
    if (haloCells < 1 || stride < 1 || grid.width == 0 || grid.height == 0) return;
    const auto pad = static_cast<std::uint32_t>(stride) * static_cast<std::uint32_t>(haloCells);
    if (iu0 >= pad) iu0 -= pad;
    else iu0 = 0;
    if (iv0 >= pad) iv0 -= pad;
    else iv0 = 0;
    const auto lastU = grid.width - 1;
    const auto lastV = grid.height - 1;
    if (lastU - iu1 >= pad) iu1 += pad;
    else iu1 = lastU;
    if (lastV - iv1 >= pad) iv1 += pad;
    else iv1 = lastV;
}

SampledWindow makeSampledWindow(const RayGrid& grid, const BoundingBox& box, int stride,
                                int haloCells)
{
    SampledWindow w;
    if (stride < 1) return w;

    std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
    if (box.valid())
    {
        if (!gridWindowFromModelAabb(grid, box, iu0, iu1, iv0, iv1)) return w;
        growGridWindowByStride(grid, stride, haloCells, iu0, iu1, iv0, iv1);
    }
    else
    {
        if (grid.width == 0 || grid.height == 0) return w;
        iu1 = grid.width - 1;
        iv1 = grid.height - 1;
    }

    w.valid = true;
    w.su0 = sampledIndex(iu0, stride);
    w.sv0 = sampledIndex(iv0, stride);
    w.su1 = sampledIndex(iu1, stride);
    w.sv1 = sampledIndex(iv1, stride);
    return w;
}

bool sampledWindowContains(const SampledWindow& w, const vsg::vec3& pos, std::size_t axis,
                           const RayGrid& grid, int stride)
{
    if (!w.valid) return false;
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const auto su = sampledIndex(grid.indexU(vecComponent(pos, u)), stride);
    const auto sv = sampledIndex(grid.indexV(vecComponent(pos, v)), stride);
    return su >= w.su0 && su <= w.su1 && sv >= w.sv0 && sv <= w.sv1;
}

void collectSectionTris(const RayModel& rayModel, int stride, const BoundingBox& box,
                        bool clipToAabb, int haloCells, std::vector<CollectedTri>& tris)
{
    if (stride < 1) return;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;

        const SampledWindow w = makeSampledWindow(*grid, box, stride, haloCells);
        if (!w.valid) continue;

        const auto sampledW = w.su1 - w.su0 + 1;
        const auto cellCount =
            static_cast<std::size_t>(sampledW) * static_cast<std::size_t>(w.sv1 - w.sv0 + 1);
        std::vector<std::vector<SectionPoint>> cells(cellCount);

        for (std::uint32_t sv = w.sv0; sv <= w.sv1; ++sv)
        {
            const auto iv = sv * static_cast<std::uint32_t>(stride);
            if (iv >= grid->height) continue;
            for (std::uint32_t su = w.su0; su <= w.su1; ++su)
            {
                const auto iu = su * static_cast<std::uint32_t>(stride);
                if (iu >= grid->width) continue;
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                collectSectionPoints(*grid, axis, iu, iv, box, clipToAabb, cells[flat]);
            }
        }

        const double du = grid->spacingU * static_cast<double>(stride);
        const double dv = grid->spacingV * static_cast<double>(stride);
        const float cellStep = static_cast<float>(std::sqrt(du * du + dv * dv));
        const float maxAlong =
            (clipToAabb ? kInspectionMaxEdgeScale : kCutFaceMaxEdgeScale) * cellStep;
        const float maxLink =
            (clipToAabb ? kInspectionMaxEdgeScale : kCutFaceMaxLinkScale) * cellStep;

        std::vector<std::array<SectionVert, 3>> quads;
        const auto axisByte = static_cast<std::uint8_t>(axis);
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
            {
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                const auto east = flat + 1;
                const auto north = flat + sampledW;
                const auto northEast = north + 1;
                const auto before = quads.size();
                emitSectionQuad(cells[flat], cells[east], cells[north], cells[northEast],
                                maxAlong, maxLink, quads);
                for (auto q = before; q < quads.size(); ++q)
                    tris.push_back({quads[q], axisByte, su, sv});
            }
        }
    }
}

} // namespace

const char* toString(PatchResult result)
{
    switch (result)
    {
    case PatchResult::Ok: return "ok";
    case PatchResult::LayoutChanged: return "layout changed";
    case PatchResult::CellTooDense: return "cell over endpoint ceiling";
    case PatchResult::OutOfSpace: return "free list exhausted";
    }
    return "unknown";
}

void GaussianSplatCache::clear()
{
    // Soft clear: keep GPU capacity/pipelines so the next rebuild stays warm.
    // After updateRegion, used slots are not a packed prefix — _live is only
    // the allocated count — so every slot in the buffer has to be zeroed.
    if (_capacity > 0)
    {
        for (std::size_t i = 0; i < _capacity; ++i)
            _set.clearSlot(i);
        _set.markDirty();
    }

    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _live = 0;
    _allocEnd = 0;
    _gpuNeedsCompile = false;
    _skipCutSplats = false;
    _capacity = _set.capacity();
    _set.setDrawCount(0);
    clearCutFace();
}

void GaussianSplatCache::release()
{
    _set.resize(0);
    _set.setOverlay(nullptr);
    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = Point3d{0.0, 0.0, 0.0};
    _stride = 0;
    _capacity = 0;
    _live = 0;
    _allocEnd = 0;
    _gpuNeedsCompile = true;
    _skipCutSplats = false;
    _cutFaceStride = 0;
    _cutFaceResolution = Point3d{0.0, 0.0, 0.0};
    _cutFaceLayout = {};
    for (auto& refs : _cutFaceRefs) refs.clear();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    _section.release();
    _inspectionSection.release();
}

bool GaussianSplatCache::layoutMatches(const RayModel& rayModel, int stride) const
{
    if (stride != _stride || rayModel.resolution() != _resolution) return false;
    if (_stride == 0 || _capacity == 0 || _set.capacity() == 0) return false;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        const AxisLayout& layout = _axes[axis];
        if (!grid)
        {
            if (layout.present) return false;
            continue;
        }
        if (!layout.present) return false;
        if (layout.width != grid->width || layout.height != grid->height) return false;
        if (layout.sampledW != sampledCount(grid->width, stride) ||
            layout.sampledH != sampledCount(grid->height, stride))
            return false;
        const auto expected =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        if (_cellRefs[axis].size() != expected) return false;
    }
    return true;
}

GaussianSplatCache::CellRef& GaussianSplatCache::cellRef(std::size_t axis,
                                                         std::uint32_t su,
                                                         std::uint32_t sv)
{
    const AxisLayout& layout = _axes[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cellRefs[axis][flat];
}

const GaussianSplatCache::CellRef& GaussianSplatCache::cellRef(std::size_t axis,
                                                               std::uint32_t su,
                                                               std::uint32_t sv) const
{
    const AxisLayout& layout = _axes[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cellRefs[axis][flat];
}

void GaussianSplatCache::clearSlots(std::uint32_t first, std::uint32_t count)
{
    for (std::uint32_t i = 0; i < count; ++i)
        _set.clearSlot(static_cast<std::size_t>(first + i));
    _set.noteDirtySlots(first, count);
}

void GaussianSplatCache::addFreeRange(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;

    FreeRange range{first, length};
    auto it = std::lower_bound(
        _freeList.begin(), _freeList.end(), range,
        [](const FreeRange& a, const FreeRange& b) { return a.first < b.first; });

    it = _freeList.insert(it, range);

    // Merge with next.
    if (it + 1 != _freeList.end() && it->first + it->length == (it + 1)->first)
    {
        it->length += (it + 1)->length;
        _freeList.erase(it + 1);
    }
    // Merge with previous.
    if (it != _freeList.begin())
    {
        auto prev = it - 1;
        if (prev->first + prev->length == it->first)
        {
            prev->length += it->length;
            _freeList.erase(it);
        }
    }
}

void GaussianSplatCache::freeBlock(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;
    clearSlots(first, length);
    addFreeRange(first, length);

    if (_live >= length) _live -= length;
    else _live = 0;
}

// Free-list only. Growing GPU capacity here would rebind the live VertexIndexDraw
// with uncompiled BufferInfos, so exhaustion has to fall back to a full rebuild.
bool GaussianSplatCache::allocBlock(std::uint32_t length, std::uint32_t* outFirst)
{
    if (length == 0 || !outFirst) return false;

    for (auto it = _freeList.begin(); it != _freeList.end(); ++it)
    {
        if (it->length < length) continue;

        const std::uint32_t first = it->first;
        if (it->length == length)
        {
            _freeList.erase(it);
        }
        else
        {
            it->first += length;
            it->length -= length;
        }
        *outFirst = first;
        _live += length;
        const std::uint32_t end = first + length;
        if (end > _allocEnd) _allocEnd = end;
        return true;
    }
    return false;
}

bool GaussianSplatCache::fillCell(const RayModel& rayModel,
                                  std::size_t axis,
                                  std::uint32_t iu,
                                  std::uint32_t iv,
                                  float radius,
                                  const SplatStyle& style,
                                  CellRef& ref,
                                  bool clearTrailing)
{
    const RayGrid* grid = rayModel.grid(axis);
    if (!grid || ref.first == CellRef::kInvalid || ref.block == 0) return true;

    const RaySlot& slot = grid->at(iu, iv);
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;
    const double u0 = grid->sampleU(iu);
    const double v0 = grid->sampleV(iv);
    const double cellDiag = std::sqrt(static_cast<double>(grid->spacingU) *
                                          static_cast<double>(grid->spacingU) +
                                      static_cast<double>(grid->spacingV) *
                                          static_cast<double>(grid->spacingV));
    const int stride = (_stride > 0) ? _stride : 1;

    vsg::vec4 stock = style.stockColor;
    stock.a = style.opacity;
    vsg::vec4 tool = style.toolColor;
    tool.a = style.opacity;

    std::uint32_t written = 0;
    if (!slot.empty())
    {
        auto spans = grid->pool.span(slot);
        for (const Interval& span : spans)
        {
            if (!span.hasSolidLength()) continue;
            Point3d start{0.0, 0.0, 0.0};
            Point3d end{0.0, 0.0, 0.0};
            start[axis] = grid->fromTick(span.begin);
            end[axis] = grid->fromTick(span.end);
            start[u] = u0;
            end[u] = u0;
            start[v] = v0;
            end[v] = v0;

            const double modelLength = end[axis] - start[axis];
            const float spanRadius = splatRadiusForSpan(radius, modelLength, cellDiag, stride);
            bool skipStart = false;
            bool skipEnd = false;
            const bool onRim = !_skipCutSplats || cutCellOnRim(*grid, iu, iv, stride);
            skipSplatEnds(span, modelLength, leftoverHideLength(*grid, stride),
                          bleedHideLength(*grid, stride), _skipCutSplats, onRim, skipStart,
                          skipEnd);
            // Cut disks are the cut surface when mesh is off — never view-cull them.
            const bool protectStart = span.cutBegin() && !_skipCutSplats;
            const bool protectEnd = span.cutEnd() && !_skipCutSplats;
            if (!protectStart || !protectEnd)
            {
                bool vs = skipStart;
                bool ve = skipEnd;
                applyViewEndPolicy(start, end, iu, iv, _viewCull, vs, ve);
                if (!protectStart) skipStart = vs;
                if (!protectEnd) skipEnd = ve;
            }

            if (!skipStart && !protectStart &&
                !endpointInViewCull(*grid, iu, iv, start[axis], _viewCull))
                skipStart = true;
            if (!skipEnd && !protectEnd &&
                !endpointInViewCull(*grid, iu, iv, end[axis], _viewCull))
                skipEnd = true;

            if (!skipStart)
            {
                if (written + 1 > ref.block) break;
                const auto base = static_cast<std::size_t>(ref.first + written);
                const vsg::vec3 nStart = endpointNormal(*grid, axis, iu, iv, stride,
                                                        span.beginNormal, true, start[axis]);
                const EdgeInfo edgeStart =
                    computeEdgeMask(*grid, axis, iu, iv, stride, true, start[axis], nStart);
                _set.set(base,
                         {vsg::vec3(static_cast<float>(start[0]),
                                    static_cast<float>(start[1]),
                                    static_cast<float>(start[2])),
                          nStart, span.cutBegin() ? tool : stock, spanRadius, edgeStart.mask,
                          edgeStart.strength});
                ++written;
            }
            if (!skipEnd)
            {
                if (written + 1 > ref.block) break;
                const auto base = static_cast<std::size_t>(ref.first + written);
                const vsg::vec3 nEnd = endpointNormal(*grid, axis, iu, iv, stride, span.endNormal,
                                                      false, end[axis]);
                const EdgeInfo edgeEnd =
                    computeEdgeMask(*grid, axis, iu, iv, stride, false, end[axis], nEnd);
                _set.set(base,
                         {vsg::vec3(static_cast<float>(end[0]),
                                    static_cast<float>(end[1]),
                                    static_cast<float>(end[2])),
                          nEnd, span.cutEnd() ? tool : stock, spanRadius, edgeEnd.mask,
                          edgeEnd.strength});
                ++written;
            }
        }
    }

    ref.count = static_cast<std::uint16_t>(written);
    // Rebuild packs into a reused GPU buffer and still issues the whole
    // block. Skipped cut-face ends (and any unused tail) must be radius 0
    // or leftover dots show through the section mesh.
    if (clearTrailing || _skipCutSplats)
    {
        for (std::uint32_t i = written; i < ref.block; ++i)
            _set.clearSlot(static_cast<std::size_t>(ref.first + i));
    }
    return true;
}

PatchResult GaussianSplatCache::updateCell(const RayModel& rayModel,
                                           std::size_t axis,
                                           std::uint32_t iu,
                                           std::uint32_t iv,
                                           float radius,
                                           const SplatStyle& style)
{
    const AxisLayout& layout = _axes[axis];
    if (!layout.present) return PatchResult::Ok;

    const RayGrid* grid = rayModel.grid(axis);
    if (!grid) return PatchResult::Ok;

    const std::uint32_t su = sampledIndex(iu, _stride);
    const std::uint32_t sv = sampledIndex(iv, _stride);
    if (su >= layout.sampledW || sv >= layout.sampledH) return PatchResult::Ok;

    CellRef& ref = cellRef(axis, su, sv);
    std::uint32_t needed = endpointNeed(*grid, iu, iv, _stride, _skipCutSplats, _viewCull);
    // Match rebuild: truncate dense cells instead of forcing a full-cache rebuild.
    if (needed > static_cast<std::uint32_t>(maxEndpointsPerCell))
        needed = static_cast<std::uint32_t>(maxEndpointsPerCell);

    if (needed == 0)
    {
        if (ref.block != 0 && ref.first != CellRef::kInvalid)
            freeBlock(ref.first, ref.block);
        ref = {};
        return PatchResult::Ok;
    }

    if (needed <= ref.block && ref.first != CellRef::kInvalid)
    {
        // Live accounting: block already counted in _live from alloc/rebuild.
        fillCell(rayModel, axis, iu, iv, radius, style, ref, true);
        _set.noteDirtySlots(ref.first, ref.block);
        return PatchResult::Ok;
    }

    // +2 endpoint slack so a later 1→2 interval split often stays in-block.
    std::uint32_t blockSize = needed + 2;
    if (blockSize > static_cast<std::uint32_t>(maxEndpointsPerCell))
        blockSize = static_cast<std::uint32_t>(maxEndpointsPerCell);

    if (ref.block != 0 && ref.first != CellRef::kInvalid)
        freeBlock(ref.first, ref.block);

    std::uint32_t first = 0;
    if (!allocBlock(blockSize, &first))
    {
        // Slack is opportunistic: an exact fit still beats a full rebuild.
        blockSize = needed;
        if (!allocBlock(blockSize, &first))
        {
            ref = {};
            return PatchResult::OutOfSpace;
        }
    }

    ref.first = first;
    ref.block = static_cast<std::uint16_t>(blockSize);
    ref.count = 0;
    if (!fillCell(rayModel, axis, iu, iv, radius, style, ref, true))
    {
        freeBlock(ref.first, ref.block);
        ref = {};
        return PatchResult::OutOfSpace;
    }
    _set.noteDirtySlots(ref.first, ref.block);
    return PatchResult::Ok;
}

vsg::ref_ptr<vsg::Node> GaussianSplatCache::rebuild(const RayModel& rayModel,
                                                    int stride,
                                                    const std::array<float, 3>& radii,
                                                    const SplatStyle& style,
                                                    bool skipCutSplats,
                                                    const SplatViewCull& viewCull)
{
    _skipCutSplats = skipCutSplats;
    _viewCull = viewCull;
    auto lock = rayModel.lockChains();

    if (rayModel.rayCount() == 0)
    {
        _set.setDrawCount(0);
        _set.markDirty();
        clearSectionGrid();
        return _set.node();
    }

    if (stride < 1) stride = 1;

    _axes = {};
    for (auto& refs : _cellRefs) refs.clear();
    _freeList.clear();
    _resolution = rayModel.resolution();
    _stride = stride;
    _live = 0;

    const BoundingBox stockBounds = rayModel.bounds();

    std::array<std::vector<std::uint32_t>, 3> needs{};
    std::array<std::vector<std::uint32_t>, 3> prefix{};
    std::size_t live = 0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;

        AxisLayout& layout = _axes[axis];
        layout.present = true;
        layout.width = grid->width;
        layout.height = grid->height;
        layout.sampledW = sampledCount(grid->width, stride);
        layout.sampledH = sampledCount(grid->height, stride);
        layout.packedBase = live;

        const auto cellCount =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        _cellRefs[axis].assign(cellCount, CellRef{});
        needs[axis].assign(cellCount, 0u);

        tbb::parallel_for(std::size_t{0}, cellCount, [&](std::size_t flat)
        {
            const auto su = flat % layout.sampledW;
            const auto sv = flat / layout.sampledW;
            const auto iu = static_cast<std::uint32_t>(su) * static_cast<std::uint32_t>(stride);
            const auto iv = static_cast<std::uint32_t>(sv) * static_cast<std::uint32_t>(stride);
            if (!cellInViewCull(*grid, iu, iv, stockBounds, viewCull) &&
                (skipCutSplats || !cellHasCutTag(*grid, iu, iv)))
            {
                needs[axis][flat] = 0;
                return;
            }
            auto n = endpointNeed(*grid, iu, iv, stride, skipCutSplats, viewCull);
            if (n > static_cast<std::uint32_t>(maxEndpointsPerCell))
                n = static_cast<std::uint32_t>(maxEndpointsPerCell);
            needs[axis][flat] = n;
        });

        prefix[axis].assign(cellCount + 1, 0u);
        for (std::size_t i = 0; i < cellCount; ++i)
            prefix[axis][i + 1] = prefix[axis][i] + needs[axis][i];

        live += prefix[axis][cellCount];
    }

    if (live == 0)
    {
        // A boolean can empty every sampled cell while the grids remain.
        // Keep the compiled buffers; issue nothing this frame.
        _set.setDrawCount(0);
        _set.markDirty();
        clearSectionGrid();
        return _set.node();
    }

    // Reuse the compiled arrays when they already hold the packed prefix.
    // Growing (or the first alloc) rebinds BufferInfos and needs compile.
    // 2x headroom is only reserved when we have to allocate anyway, so a
    // later patch can grow cells without a new GPU buffer.
    const std::size_t capBefore = _set.capacity();
    const bool hadNode = _set.node() != nullptr;
    if (capBefore < live)
        _set.ensureCapacity(live * 2);
    // Capacity growth rebinds BufferInfos (needs compile). Also honour any
    // needsCompile already set on the set/section overlays.
    _gpuNeedsCompile = !hadNode || _set.capacity() != capBefore || _set.needsCompile() ||
                       _section.needsCompile() || _inspectionSection.needsCompile();
    _capacity = _set.capacity();
    _live = live;
    _allocEnd = 0;
    if (live > 0)
        _allocEnd = static_cast<std::uint32_t>(live);

    // Reused arrays still hold the previous packed/patched discs. Zero them
    // before refill so skipped leftovers cannot linger in old GPU slots.
    for (std::size_t i = 0; i < _capacity; ++i)
        _set.clearSlot(i);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const float radius = radii[axis];
        const AxisLayout& layout = _axes[axis];
        const auto cellCount =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        const auto axisBase = static_cast<std::uint32_t>(layout.packedBase);

        tbb::parallel_for(std::size_t{0}, cellCount, [&](std::size_t flat)
        {
            const auto n = needs[axis][flat];
            if (n == 0) return;

            const auto su = flat % layout.sampledW;
            const auto sv = flat / layout.sampledW;
            const auto iu = static_cast<std::uint32_t>(su) * static_cast<std::uint32_t>(stride);
            const auto iv = static_cast<std::uint32_t>(sv) * static_cast<std::uint32_t>(stride);

            CellRef& ref = _cellRefs[axis][flat];
            ref.first = axisBase + prefix[axis][flat];
            ref.block = static_cast<std::uint16_t>(n);
            ref.count = 0;
            fillCell(rayModel, axis, iu, iv, radius, style, ref, true);
        });
    }

    // Unused tail is not submitted. Do not walk it: the draw count is `live`.
    if (_capacity > live)
    {
        _freeList.push_back(FreeRange{static_cast<std::uint32_t>(live),
                                      static_cast<std::uint32_t>(_capacity - live)});
    }

    _set.setDrawCount(live);
    _set.markDirty();
    presentCutFace();
    return _set.node();
}

PatchResult GaussianSplatCache::updateRegion(const RayModel& rayModel,
                                             const BoundingBox& modelAabb,
                                             int stride,
                                             const std::array<float, 3>& radii,
                                             const SplatStyle& style,
                                             bool skipCutSplats,
                                             const SplatViewCull& viewCull)
{
    if (!layoutMatches(rayModel, stride)) return PatchResult::LayoutChanged;
    if (!modelAabb.valid()) return PatchResult::LayoutChanged;
    _skipCutSplats = skipCutSplats;
    _viewCull = viewCull;

    auto lock = rayModel.lockChains();
    const BoundingBox stockBounds = rayModel.bounds();

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_axes[axis].present) continue;
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid) continue;

        std::uint32_t iu0 = 0, iu1 = 0, iv0 = 0, iv1 = 0;
        if (!gridWindowFromModelAabb(*grid, modelAabb, iu0, iu1, iv0, iv1)) continue;
        growGridWindowByStride(*grid, _stride, kCutFaceHaloCells, iu0, iu1, iv0, iv1);

        const float radius = radii[axis];
        for (std::uint32_t iv = iv0; iv <= iv1; ++iv)
        {
            if (static_cast<int>(iv) % _stride != 0) continue;
            for (std::uint32_t iu = iu0; iu <= iu1; ++iu)
            {
                if (static_cast<int>(iu) % _stride != 0) continue;
                if (!cellInViewCull(*grid, iu, iv, stockBounds, viewCull) &&
                    (skipCutSplats || !cellHasCutTag(*grid, iu, iv)))
                {
                    const auto su = iu / static_cast<std::uint32_t>(_stride);
                    const auto sv = iv / static_cast<std::uint32_t>(_stride);
                    if (su < _axes[axis].sampledW && sv < _axes[axis].sampledH)
                    {
                        CellRef& ref = cellRef(axis, su, sv);
                        if (!ref.empty())
                        {
                            clearSlots(ref.first, ref.count);
                            _set.noteDirtySlots(ref.first, ref.block);
                            freeBlock(ref.first, ref.block);
                            ref = {};
                        }
                    }
                    continue;
                }
                const PatchResult cell = updateCell(rayModel, axis, iu, iv, radius, style);
                if (cell != PatchResult::Ok) return cell;
            }
        }
    }

    // Allocated slots can sit past the last packed prefix. Draw through
    // _allocEnd only: the unused tail is not submitted, even if leftover
    // radius is still in memory. Freed holes below _allocEnd were cleared.
    // CPU already rewrote only the dirty-window cells. Copy those slot
    // ranges; do not dirty() the whole set.
    _set.setDrawCount(_allocEnd);
    _set.flushDirty();
    return PatchResult::Ok;
}

void GaussianSplatCache::markDirty()
{
    _set.markDirty();
    _section.markDirty();
    _inspectionSection.markDirty();
}

void GaussianSplatCache::noteCompiled()
{
    _gpuNeedsCompile = false;
    _set.noteCompiled();
    _section.noteCompiled();
    _inspectionSection.noteCompiled();
}

void GaussianSplatCache::clearSectionGrid()
{
    _inspectionSection.setDrawCount(0);
    _inspectionSection.markDirty();
    presentCutFace();
}

void GaussianSplatCache::clearCutFace()
{
    _cutFaceStride = 0;
    _cutFaceResolution = Point3d{0.0, 0.0, 0.0};
    _cutFaceLayout = {};
    for (auto& refs : _cutFaceRefs) refs.clear();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    _section.setDrawCount(0);
    _section.markDirty();
}

void GaussianSplatCache::presentCutFace()
{
    if (_cutFaceLive > 0 && _section.node())
        _set.setOverlay(_section.node());
    else
        _set.setOverlay(nullptr);
}

void GaussianSplatCache::uploadSectionTris(SectionLineSet& dest, const std::vector<OverlayTri>& tris,
                                           const vsg::vec4& color)
{
    if (tris.empty())
    {
        dest.setDrawCount(0);
        dest.markDirty();
        presentCutFace();
        return;
    }

    dest.ensureCapacity(tris.size());
    for (std::size_t i = 0; i < tris.size(); ++i)
        dest.setTriangle(i, tris[i].v[0].pos, tris[i].v[1].pos, tris[i].v[2].pos,
                         tris[i].v[0].normal, tris[i].v[1].normal, tris[i].v[2].normal, color);
    dest.setDrawCount(tris.size());
    dest.markDirty();
    _set.setOverlay(dest.node());
}

void GaussianSplatCache::appendCutFaceTris(const RayModel& rayModel, int stride,
                                           const BoundingBox& box,
                                           std::vector<OverlayTri>& out, bool clipToAabb,
                                           int haloCells) const
{
    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, box, clipToAabb, haloCells, collected);
    out.reserve(out.size() + collected.size());
    for (const auto& t : collected)
    {
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        out.push_back(tri);
    }
}

bool GaussianSplatCache::cutFaceLayoutMatches(const RayModel& rayModel, int stride) const
{
    if (stride < 1 || stride != _cutFaceStride) return false;
    if (rayModel.resolution() != _cutFaceResolution) return false;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        const CutFaceLayout& layout = _cutFaceLayout[axis];
        if (!grid || grid->empty())
        {
            if (layout.present) return false;
            continue;
        }
        if (!layout.present) return false;
        if (layout.sampledW != sampledCount(grid->width, stride) ||
            layout.sampledH != sampledCount(grid->height, stride))
            return false;
        const auto expected =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        if (_cutFaceRefs[axis].size() != expected) return false;
    }
    return true;
}

void GaussianSplatCache::ensureCutFaceLayout(const RayModel& rayModel, int stride)
{
    _cutFaceStride = stride;
    _cutFaceResolution = rayModel.resolution();
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        CutFaceLayout& layout = _cutFaceLayout[axis];
        if (!grid || grid->empty())
        {
            layout = {};
            _cutFaceRefs[axis].clear();
            continue;
        }
        layout.present = true;
        layout.sampledW = sampledCount(grid->width, stride);
        layout.sampledH = sampledCount(grid->height, stride);
        const auto n =
            static_cast<std::size_t>(layout.sampledW) * static_cast<std::size_t>(layout.sampledH);
        _cutFaceRefs[axis].assign(n, CellRef{});
    }
}

GaussianSplatCache::CellRef& GaussianSplatCache::cutFaceRef(std::size_t axis, std::uint32_t su,
                                                            std::uint32_t sv)
{
    const CutFaceLayout& layout = _cutFaceLayout[axis];
    const auto flat =
        static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
    return _cutFaceRefs[axis][flat];
}

void GaussianSplatCache::addCutFaceFreeRange(std::uint32_t first, std::uint32_t length)
{
    if (length == 0) return;
    FreeRange range{first, length};
    auto it = std::lower_bound(
        _cutFaceFree.begin(), _cutFaceFree.end(), range,
        [](const FreeRange& a, const FreeRange& b) { return a.first < b.first; });
    it = _cutFaceFree.insert(it, range);
    if (it + 1 != _cutFaceFree.end() && it->first + it->length == (it + 1)->first)
    {
        it->length += (it + 1)->length;
        _cutFaceFree.erase(it + 1);
    }
    if (it != _cutFaceFree.begin())
    {
        auto prev = it - 1;
        if (prev->first + prev->length == it->first)
        {
            prev->length += it->length;
            _cutFaceFree.erase(it);
        }
    }
}

bool GaussianSplatCache::allocCutFaceBlock(std::uint32_t length, std::uint32_t* outFirst)
{
    if (length == 0 || !outFirst) return false;

    for (auto it = _cutFaceFree.begin(); it != _cutFaceFree.end(); ++it)
    {
        if (it->length < length) continue;
        const std::uint32_t first = it->first;
        if (it->length == length) _cutFaceFree.erase(it);
        else
        {
            it->first += length;
            it->length -= length;
        }
        *outFirst = first;
        return true;
    }

    _section.ensureCapacity(static_cast<std::size_t>(_cutFaceAllocEnd) + length);
    if (static_cast<std::size_t>(_cutFaceAllocEnd) + length > _section.capacity()) return false;
    *outFirst = _cutFaceAllocEnd;
    _cutFaceAllocEnd += length;
    return true;
}

void GaussianSplatCache::freeCutFaceCell(CellRef& ref)
{
    if (ref.first == CellRef::kInvalid || ref.block == 0)
    {
        ref = {};
        return;
    }
    for (std::uint32_t i = 0; i < ref.block; ++i)
        _section.clearTriangle(static_cast<std::size_t>(ref.first + i));
    addCutFaceFreeRange(ref.first, ref.block);
    if (_cutFaceLive >= ref.count) _cutFaceLive -= ref.count;
    else _cutFaceLive = 0;
    ref = {};
}

bool GaussianSplatCache::writeCutFaceCell(std::size_t axis, std::uint32_t su, std::uint32_t sv,
                                          const std::vector<OverlayTri>& tris)
{
    CellRef& ref = cutFaceRef(axis, su, sv);
    freeCutFaceCell(ref);
    if (tris.empty()) return true;
    // Dense cells used to return false → patchCutFace full-rebuild (minutes).
    // Keep a capped local surface instead of freezing the UI.
    const auto n = static_cast<std::uint32_t>(
        std::min(tris.size(), static_cast<std::size_t>(maxCutFaceTrisPerCell)));
    std::uint32_t first = 0;
    if (!allocCutFaceBlock(n, &first)) return false;
    ref.first = first;
    ref.block = static_cast<std::uint16_t>(n);
    ref.count = static_cast<std::uint16_t>(n);
    for (std::uint32_t i = 0; i < n; ++i)
    {
        const OverlayTri& tri = tris[i];
        _section.setTriangle(static_cast<std::size_t>(first + i), tri.v[0].pos, tri.v[1].pos,
                             tri.v[2].pos, tri.v[0].normal, tri.v[1].normal, tri.v[2].normal,
                             _cutFaceColor);
    }
    _cutFaceLive += n;
    return true;
}

void GaussianSplatCache::updateSectionGrid(const RayModel& rayModel, int stride,
                                           const BoundingBox& sectionAabb,
                                           const vsg::vec4& color)
{
    if (!sectionAabb.valid() || stride < 1)
    {
        clearSectionGrid();
        return;
    }

    auto lock = rayModel.lockChains();
    std::vector<OverlayTri> tris;
    appendCutFaceTris(rayModel, stride, sectionAabb, tris, false, kCutFaceHaloCells);
    uploadSectionTris(_inspectionSection, tris, color);
}

void GaussianSplatCache::rebuildCutFace(const RayModel& rayModel, int stride,
                                        const vsg::vec4& color)
{
    _cutFaceColor = color;
    if (stride < 1)
    {
        clearCutFace();
        presentCutFace();
        return;
    }

    auto lock = rayModel.lockChains();
    _cutFaceFree.clear();
    _cutFaceAllocEnd = 0;
    _cutFaceLive = 0;
    ensureCutFaceLayout(rayModel, stride);

    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, BoundingBox{}, false, 0, collected);

    std::array<std::vector<std::vector<OverlayTri>>, 3> buckets;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_cutFaceLayout[axis].present) continue;
        buckets[axis].resize(_cutFaceRefs[axis].size());
    }

    for (const CollectedTri& t : collected)
    {
        if (t.axis > 2 || !_cutFaceLayout[t.axis].present) continue;
        const CutFaceLayout& layout = _cutFaceLayout[t.axis];
        if (t.su >= layout.sampledW || t.sv >= layout.sampledH) continue;
        const auto flat =
            static_cast<std::size_t>(t.sv) * layout.sampledW + static_cast<std::size_t>(t.su);
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        buckets[t.axis][flat].push_back(tri);
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!_cutFaceLayout[axis].present) continue;
        const CutFaceLayout& layout = _cutFaceLayout[axis];
        for (std::uint32_t sv = 0; sv < layout.sampledH; ++sv)
        {
            for (std::uint32_t su = 0; su < layout.sampledW; ++su)
            {
                const auto flat =
                    static_cast<std::size_t>(sv) * layout.sampledW + static_cast<std::size_t>(su);
                if (buckets[axis][flat].empty()) continue;
                if (!writeCutFaceCell(axis, su, sv, buckets[axis][flat]))
                {
                    // Rare: a single cell exceeded the soft cap. Keep what we have.
                    break;
                }
            }
        }
    }

    _section.setDrawCount(_cutFaceAllocEnd);
    _section.markDirty();
    presentCutFace();
}

void GaussianSplatCache::patchCutFace(const RayModel& rayModel, int stride,
                                      const BoundingBox& dirtyModelAabb, const vsg::vec4& color)
{
    if (!dirtyModelAabb.valid() || stride < 1) return;
    _cutFaceColor = color;
    if (!cutFaceLayoutMatches(rayModel, stride))
    {
        rebuildCutFace(rayModel, stride, color);
        return;
    }

    auto lock = rayModel.lockChains();
    std::array<SampledWindow, 3> windows{};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const RayGrid* grid = rayModel.grid(axis);
        if (!grid || grid->empty()) continue;
        windows[axis] = makeSampledWindow(*grid, dirtyModelAabb, stride, kCutFaceHaloCells);
        if (!windows[axis].valid) continue;
        const SampledWindow& w = windows[axis];
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
                freeCutFaceCell(cutFaceRef(axis, su, sv));
        }
    }

    std::vector<CollectedTri> collected;
    collectSectionTris(rayModel, stride, dirtyModelAabb, false, kCutFaceHaloCells, collected);

    std::array<std::vector<std::vector<OverlayTri>>, 3> buckets;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!windows[axis].valid || !_cutFaceLayout[axis].present) continue;
        const SampledWindow& w = windows[axis];
        const auto sampledW = w.su1 - w.su0;
        const auto sampledH = w.sv1 - w.sv0;
        if (sampledW == 0 || sampledH == 0) continue;
        buckets[axis].resize(static_cast<std::size_t>(sampledW) * static_cast<std::size_t>(sampledH));
    }

    for (const CollectedTri& t : collected)
    {
        if (t.axis > 2 || !windows[t.axis].valid) continue;
        const SampledWindow& w = windows[t.axis];
        if (t.su < w.su0 || t.su >= w.su1 || t.sv < w.sv0 || t.sv >= w.sv1) continue;
        const auto sampledW = w.su1 - w.su0;
        const auto flat = static_cast<std::size_t>(t.sv - w.sv0) * sampledW +
                          static_cast<std::size_t>(t.su - w.su0);
        OverlayTri tri;
        tri.v[0] = {t.verts[0].pos, t.verts[0].normal};
        tri.v[1] = {t.verts[1].pos, t.verts[1].normal};
        tri.v[2] = {t.verts[2].pos, t.verts[2].normal};
        tri.axis = t.axis;
        buckets[t.axis][flat].push_back(tri);
    }

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!windows[axis].valid || buckets[axis].empty()) continue;
        const SampledWindow& w = windows[axis];
        const auto sampledW = w.su1 - w.su0;
        for (std::uint32_t sv = w.sv0; sv < w.sv1; ++sv)
        {
            for (std::uint32_t su = w.su0; su < w.su1; ++su)
            {
                const auto flat = static_cast<std::size_t>(sv - w.sv0) * sampledW +
                                  static_cast<std::size_t>(su - w.su0);
                if (!writeCutFaceCell(axis, su, sv, buckets[axis][flat]))
                {
                    rebuildCutFace(rayModel, stride, color);
                    return;
                }
            }
        }
    }

    _section.setDrawCount(_cutFaceAllocEnd);
    _section.flushDirty();
    presentCutFace();
}

void GaussianSplatCache::showCutFace()
{
    presentCutFace();
}

void GaussianSplatCache::restoreCutFace(const RayModel& rayModel, int stride,
                                        const vsg::vec4& color)
{
    _cutFaceColor = color;
    if (cutFaceLayoutMatches(rayModel, stride))
    {
        showCutFace();
        return;
    }
    rebuildCutFace(rayModel, stride, color);
}

} // namespace app
