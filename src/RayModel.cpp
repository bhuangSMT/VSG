#include "RayModel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "BRep.h"

namespace app
{

namespace
{

// The one ceiling on how fine a cast may be, and it exists only because the
// spans have to fit in memory. The UI lets the resolution be typed in, so it
// has to be enforced rather than assumed.
//
// It is emphatically not a ceiling on what can be drawn. The renderer thins a
// fine model out by grid layers, so a resolution far past anything the GPU
// could hold is still worth casting and is still displayed; the model keeps
// every ray it cast. Setting this by what looks good on screen would throw away
// the accuracy the fine cast exists for.
//
// Eight million casts in a direction is roughly 2.5 GB of spans on a part that
// the rays cross a couple of times, which is what a 24 GB machine can give up
// without paging. Around 2800 steps across the bounding box.
constexpr double maxStepsPerAxis = 8192.0;
constexpr double maxCastsPerDirection = 8.0e6;

// What one casting direction costs: the steps along the two axes its grid runs
// over, and how many casts that comes to.
struct CastExtent
{
    double uSteps = 0.0;
    double vSteps = 0.0;
    double casts = 0.0;
};

CastExtent castExtent(const BoundingBox& bounds, const Point3d& resolution, std::size_t axis)
{
    const std::size_t u = (axis + 1) % 3;
    const std::size_t v = (axis + 2) % 3;

    CastExtent extent;
    extent.uSteps = std::floor(bounds.extent(u) / resolution[u]);
    extent.vSteps = std::floor(bounds.extent(v) / resolution[v]);
    extent.casts = (extent.uSteps + 1.0) * (extent.vSteps + 1.0);
    return extent;
}

bool withinCastBudget(const BoundingBox& bounds, const Point3d& resolution)
{
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const CastExtent extent = castExtent(bounds, resolution, axis);
        if (extent.uSteps > maxStepsPerAxis || extent.vSteps > maxStepsPerAxis ||
            extent.casts > maxCastsPerDirection)
            return false;
    }
    return true;
}

Point3d toPoint(const vsg::vec3& v)
{
    return Point3d{static_cast<double>(v.x),
                   static_cast<double>(v.y),
                   static_cast<double>(v.z)};
}

// One crossing of the surface: where along the cast axis it happened, and the
// normal of the face that was crossed.
struct Hit
{
    double along = 0.0;
    Normal3f normal{0.0f, 0.0f, 0.0f};
};

// Collect where the line parallel to `axis` through (u0, v0) crosses the
// surface, as coordinates along `axis`. u and v are the two remaining axes.
//
// Because the ray is axis aligned, the test reduces to a point-in-triangle
// check in the (u, v) plane; the crossing coordinate is then the barycentric
// blend of the triangle's own `axis` coordinates.
//
// The BRep's BVH supplies the candidates, so only the triangles whose footprint
// straddles (u0, v0) are looked at rather than the whole mesh.
void collectHits(const BRep& brep,
                 std::size_t axis, std::size_t u, std::size_t v,
                 double u0, double v0,
                 std::vector<Hit>& hits)
{
    hits.clear();

    const auto& verts = brep.vertices();
    const auto& faceOffsets = brep.faceOffsets();
    const auto& faceVertices = brep.faceVertices();

    const auto test = [&](std::size_t f) {
        const std::uint32_t begin = faceOffsets[f];
        if (faceOffsets[f + 1] - begin < 3) return;

        const Point3d a = toPoint(verts[faceVertices[begin]]);
        const Point3d b = toPoint(verts[faceVertices[begin + 1]]);
        const Point3d c = toPoint(verts[faceVertices[begin + 2]]);

        const double au = a[u], av = a[v];
        const double bu = b[u], bv = b[v];
        const double cu = c[u], cv = c[v];

        // Cheap reject: the ray cannot cross a triangle whose footprint in the
        // (u, v) plane does not contain the sample point. The BVH prunes whole
        // groups of triangles on the same grounds, but the ones it hands back
        // still have to be checked individually.
        const double loU = std::min({au, bu, cu});
        const double hiU = std::max({au, bu, cu});
        const double loV = std::min({av, bv, cv});
        const double hiV = std::max({av, bv, cv});
        if (u0 < loU || u0 > hiU || v0 < loV || v0 > hiV) return;

        const double denom = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);

        // The denominator is twice the projected area. When it vanishes the
        // triangle is edge-on to the ray, projecting to a line that cannot be
        // crossed cleanly. Compare against the footprint so the test holds at
        // any model scale.
        const double areaScale = (hiU - loU) * (hiV - loV);
        if (std::abs(denom) <= 1e-12 * std::max(areaScale, 1e-300)) return;

        const double w0 = ((bv - cv) * (u0 - cu) + (cu - bu) * (v0 - cv)) / denom;
        const double w1 = ((cv - av) * (u0 - cu) + (au - cu) * (v0 - cv)) / denom;
        const double w2 = 1.0 - w0 - w1;

        // Hits exactly on an edge or vertex are kept; duplicates from the
        // neighbouring triangles are merged by the caller.
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0) return;

        Hit hit;
        hit.along = w0 * a[axis] + w1 * b[axis] + w2 * c[axis];

        // Face normal from the edge cross product. Its sign follows the
        // triangle's winding, which an STL is under no obligation to get
        // consistently right, so the caller re-orients it against the
        // direction of travel instead of trusting it.
        const double e1[3]{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
        const double e2[3]{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
        const double n[3]{e1[1] * e2[2] - e1[2] * e2[1],
                          e1[2] * e2[0] - e1[0] * e2[2],
                          e1[0] * e2[1] - e1[1] * e2[0]};

        const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (length > 0.0)
        {
            for (int i = 0; i < 3; ++i)
                hit.normal[i] = static_cast<float>(n[i] / length);
        }

        hits.push_back(hit);
    };

    const BVH& bvh = brep.bvh();
    if (bvh.empty())
    {
        // A BRep built without a hierarchy still has to cast correctly.
        const std::size_t faces = brep.faceCount();
        for (std::size_t f = 0; f < faces; ++f) test(f);
        return;
    }

    bvh.query(u, v, u0, v0, test);
}

// Sort the crossings and drop repeats, which arise whenever a ray meets an edge
// or a vertex shared by several triangles. Of a merged group the first crossing
// is the one kept, so a shared edge takes the normal of one of its faces rather
// than an average of them.
void sortAndMerge(std::vector<Hit>& hits, double tolerance)
{
    std::sort(hits.begin(), hits.end(),
              [](const Hit& lhs, const Hit& rhs) { return lhs.along < rhs.along; });

    auto last = std::unique(hits.begin(), hits.end(),
                            [tolerance](const Hit& lhs, const Hit& rhs) {
                                return std::abs(rhs.along - lhs.along) <= tolerance;
                            });
    hits.erase(last, hits.end());
}

Point3d makePoint(std::size_t axis, std::size_t u, std::size_t v,
                  double along, double u0, double v0)
{
    Point3d p{0.0, 0.0, 0.0};
    p[axis] = along;
    p[u] = u0;
    p[v] = v0;
    return p;
}

// Point a face normal out of the solid. The ray travels up the axis, so at an
// entry the outward normal opposes it and at an exit it agrees. `entering`
// picks which. This makes the result independent of the triangle winding, which
// an STL may well have inconsistent.
Normal3f orientOutward(Normal3f normal, std::size_t axis, bool entering)
{
    const float along = normal[axis];
    const bool flip = entering ? (along > 0.0f) : (along < 0.0f);
    if (flip)
    {
        for (int i = 0; i < 3; ++i) normal[i] = -normal[i];
    }
    return normal;
}

} // namespace

namespace
{

Point3d finestWithin(const BoundingBox& bounds, const Point3d& requested)
{
    if (!bounds.valid() || withinCastBudget(bounds, requested)) return requested;

    // Casts in a direction go as the product of its two step counts, so
    // coarsening everything by k divides them by k squared. That gives a
    // starting factor directly; the step limit is linear in k.
    double scale = 1.0;
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const CastExtent extent = castExtent(bounds, requested, axis);

        if (extent.casts > maxCastsPerDirection)
            scale = std::max(scale, std::sqrt(extent.casts / maxCastsPerDirection));

        scale = std::max(scale, extent.uSteps / maxStepsPerAxis);
        scale = std::max(scale, extent.vSteps / maxStepsPerAxis);
    }

    // Each step count is rounded down before the casts are counted, so the
    // factor above can land a shade under what is needed. Creep up until the
    // real test agrees rather than trying to invert the rounding.
    Point3d resolution = requested;
    for (int attempt = 0; attempt < 128; ++attempt)
    {
        for (std::size_t i = 0; i < 3; ++i) resolution[i] = requested[i] * scale;

        if (withinCastBudget(bounds, resolution)) break;
        scale *= 1.02;
    }

    return resolution;
}

} // namespace

Point3d RayModel::finestCastableResolution(const BRep& brep, const Point3d& requested)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (!(requested[i] > 0.0))
            throw std::invalid_argument("RayModel resolution must be positive along every axis.");
    }

    return finestWithin(BoundingBox::fromBRep(brep), requested);
}

RayModel RayModel::fromBRep(const BRep& brep, const Point3d& resolution)
{
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (!(resolution[i] > 0.0))
            throw std::invalid_argument("RayModel resolution must be positive along every axis.");
    }

    RayModel model;
    model._resolution = resolution;
    model._bounds = BoundingBox::fromBRep(brep);

    if (!model._bounds.valid() || brep.faceCount() == 0) return model;

    const Point3d& lo = model._bounds.min();
    const double tolerance = 1e-9 * std::max(model._bounds.diagonal(), 1.0);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        // The two axes the sampling grid runs over, in cyclic order: casting
        // along x steps over y then z, along y over z then x, along z over x
        // then y.
        const std::size_t u = (axis + 1) % 3;
        const std::size_t v = (axis + 2) % 3;

        const double du = resolution[u];
        const double dv = resolution[v];

        const CastExtent extent = castExtent(model._bounds, resolution, axis);
        const double uSteps = extent.uSteps;
        const double vSteps = extent.vSteps;

        // The only ceiling, and it is about memory rather than about what can
        // be drawn. Nothing is substituted; the message says what would fit and
        // leaves the choice with the caller.
        if (!withinCastBudget(model._bounds, resolution))
        {
            const Point3d finest = finestWithin(model._bounds, resolution);

            throw std::invalid_argument(
                "The ray resolution is too fine to cast for this model: it would need " +
                std::to_string(static_cast<long long>(extent.casts)) +
                " casts in one direction, and the limit is " +
                std::to_string(static_cast<long long>(maxCastsPerDirection)) +
                ". The finest that will cast is " + std::to_string(finest[0]) + ", " +
                std::to_string(finest[1]) + ", " + std::to_string(finest[2]) + ".");
        }

        const int uCount = (uSteps > 0.0) ? static_cast<int>(uSteps) : 0;
        const int vCount = (vSteps > 0.0) ? static_cast<int>(vSteps) : 0;

        // The two nested sweeps are flattened into one range so that the work
        // still divides evenly when one of the axes has only a few steps. The
        // guard above caps this at maxCastsPerDirection, so it cannot overflow.
        const int uPoints = uCount + 1;
        const int castCount = uPoints * (vCount + 1);

        // Casts are independent and complete out of order, so each chain is
        // tagged with the cast that produced it and grid order is restored
        // below. Collecting them as they finish instead would leave the chain
        // order varying from run to run.
        std::vector<std::pair<int, RayChain>> found;
        std::mutex foundMutex;

        tbb::parallel_for(
            tbb::blocked_range<int>(0, castCount),
            [&](const tbb::blocked_range<int>& range) {
                // Both buffers live for the whole range rather than per cast,
                // so the casts in a range share one set of allocations.
                std::vector<Hit> hits;
                std::vector<std::pair<int, RayChain>> localFound;

                for (int cast = range.begin(); cast != range.end(); ++cast)
                {
                    // The second axis is the slower one: for rays along x, y
                    // runs from minY to maxY before z is stepped on.
                    const int vi = cast / uPoints;
                    const int ui = cast - vi * uPoints;

                    const double v0 = lo[v] + dv * vi;
                    const double u0 = lo[u] + du * ui;

                    collectHits(brep, axis, u, v, u0, v0, hits);
                    if (hits.size() < 2) continue;

                    sortAndMerge(hits, tolerance);

                    RayChain chain;
                    chain.u = ui;
                    chain.v = vi;
                    chain.rays.reserve(hits.size() / 2);

                    // Pair the crossings: in at the first, out at the second,
                    // back in at the third, and so on. A trailing unpaired
                    // crossing is a grazing hit and cannot form a span.
                    for (std::size_t h = 0; h + 1 < hits.size(); h += 2)
                    {
                        Ray ray;
                        ray.startPoint = makePoint(axis, u, v, hits[h].along, u0, v0);
                        ray.endPoint = makePoint(axis, u, v, hits[h + 1].along, u0, v0);
                        ray.startNormal = orientOutward(hits[h].normal, axis, true);
                        ray.endNormal = orientOutward(hits[h + 1].normal, axis, false);
                        chain.rays.push_back(ray);
                    }

                    if (!chain.rays.empty()) localFound.emplace_back(cast, std::move(chain));
                }

                if (localFound.empty()) return;

                // Published once per range, not once per cast.
                const std::lock_guard<std::mutex> guard(foundMutex);
                found.insert(found.end(),
                             std::make_move_iterator(localFound.begin()),
                             std::make_move_iterator(localFound.end()));
            });

        std::sort(found.begin(), found.end(),
                  [](const std::pair<int, RayChain>& lhs, const std::pair<int, RayChain>& rhs) {
                      return lhs.first < rhs.first;
                  });

        auto& chains = model._chains[axis];
        chains.reserve(found.size());
        for (auto& entry : found) chains.push_back(std::move(entry.second));
    }

    return model;
}

RayModel::RayModel(RayModel&& other) noexcept :
    _chains(std::move(other._chains)),
    _bounds(other._bounds),
    _resolution(other._resolution)
{
}

RayModel& RayModel::operator=(RayModel&& other) noexcept
{
    if (this == &other) return *this;
    std::lock_guard<std::recursive_mutex> lock(_chainMutex);
    _chains = std::move(other._chains);
    _bounds = other._bounds;
    _resolution = other._resolution;
    return *this;
}

std::unique_lock<std::recursive_mutex> RayModel::lockChains() const
{
    return std::unique_lock<std::recursive_mutex>(_chainMutex);
}

void RayModel::removeDegenerateRays()
{
    std::lock_guard<std::recursive_mutex> lock(_chainMutex);

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double minLength = _resolution[axis];
        auto& chains = _chains[axis];

        for (auto& chain : chains)
        {
            auto& rays = chain.rays;
            rays.erase(std::remove_if(rays.begin(), rays.end(),
                                      [axis, minLength](const Ray& ray) {
                                          const double length =
                                              std::abs(ray.endPoint[axis] - ray.startPoint[axis]);
                                          return length < minLength;
                                      }),
                       rays.end());
        }

        chains.erase(std::remove_if(chains.begin(), chains.end(),
                                    [](const RayChain& chain) { return chain.rays.empty(); }),
                     chains.end());
    }
}

std::size_t RayModel::chainCount() const
{
    auto lock = lockChains();
    std::size_t total = 0;
    for (const auto& perAxis : _chains) total += perAxis.size();
    return total;
}

std::size_t RayModel::rayCount() const
{
    auto lock = lockChains();
    std::size_t total = 0;
    for (const auto& perAxis : _chains)
    {
        for (const auto& chain : perAxis) total += chain.rays.size();
    }
    return total;
}

int RayModel::strideForRayBudget(std::size_t maxRays) const
{
    if (maxRays == 0) return 1;

    const std::size_t total = rayCount();
    if (total <= maxRays) return 1;

    // Keeping every stride'th line in both grid directions leaves roughly one
    // ray in stride squared, which gives a starting point. It is a starting
    // point rather than the answer because which lines actually meet the model
    // depends on how it sits on the grid, so the real count decides.
    //
    // The search steps up one at a time instead of doubling: the drawn density
    // then lands just under the budget whatever the cast resolution, so asking
    // for a finer cast never comes back as a coarser picture.
    const double estimate = std::sqrt(static_cast<double>(total) / static_cast<double>(maxRays));

    int stride = (estimate > 1.0) ? static_cast<int>(estimate) : 1;

    while (stride < maxStride && rayCountAtStride(stride) > maxRays) ++stride;

    return stride;
}

std::size_t RayModel::rayCountAtStride(int stride) const
{
    if (stride <= 1) return rayCount();

    auto lock = lockChains();
    std::size_t total = 0;
    for (const auto& perAxis : _chains)
    {
        for (const auto& chain : perAxis)
        {
            if (chain.u % stride == 0 && chain.v % stride == 0) total += chain.rays.size();
        }
    }
    return total;
}

} // namespace app
