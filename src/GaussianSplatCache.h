// GaussianSplatCache - fixed-capacity per-cell GPU splat buffers for Ray-GS.
//
// Each sampled grid cell reserves maxEndpointsPerCell slots so a boolean AABB
// window can overwrite in place without reallocating the whole mesh.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <vsg/all.h>

#include "BoundingBox.h"
#include "GaussianSplat.h"
#include "RayModel.h"

namespace app
{

struct SplatStyle
{
    vsg::vec4 stockColor{0.48f, 0.50f, 0.54f, 1.0f};
    vsg::vec4 toolColor{0.95f, 0.35f, 0.10f, 1.0f};
    float opacity = 0.5f;
};

class GaussianSplatCache
{
public:
    static constexpr int maxIntervalsPerCell = 8;
    static constexpr int maxEndpointsPerCell = maxIntervalsPerCell * 2;

    void clear();
    bool empty() const { return _set.empty(); }

    // Full rebuild from the displayed RayModel. Returns the drawable (unfitted).
    // Throws if the model has no intervals.
    vsg::ref_ptr<vsg::Node> rebuild(const RayModel& rayModel,
                                    int stride,
                                    const std::array<float, 3>& radii,
                                    const SplatStyle& style);

    // Regenerate only cells overlapping modelAabb. Returns false when layout
    // no longer matches (caller should rebuild) or a cell exceeds capacity.
    bool updateRegion(const RayModel& rayModel,
                      const BoundingBox& modelAabb,
                      int stride,
                      const std::array<float, 3>& radii,
                      const SplatStyle& style);

    vsg::ref_ptr<vsg::Node> node() const { return _set.node(); }
    void markDirty() { _set.markDirty(); }

private:
    struct AxisLayout
    {
        bool present = false;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t sampledW = 0;
        std::uint32_t sampledH = 0;
        std::size_t baseSplat = 0;
    };

    bool layoutMatches(const RayModel& rayModel, int stride) const;
    void writeCell(const RayModel& rayModel,
                   std::size_t axis,
                   std::uint32_t iu,
                   std::uint32_t iv,
                   float radius,
                   const SplatStyle& style,
                   bool* overflow);

    GaussianSplatSet _set;
    std::array<AxisLayout, 3> _axes{};
    Point3d _resolution{0.0, 0.0, 0.0};
    int _stride = 0;
    std::size_t _capacity = 0;
};

} // namespace app
