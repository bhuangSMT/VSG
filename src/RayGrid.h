// RayGrid - dense fixed grid of axis-aligned stock rays with int32 tick intervals.
//
// Each cast direction owns its own RayGrid + IntervalPool. Grids are independent
// so any axis can be dropped later without schema changes. Layout comes from the
// BRep AABB + resolution; empty casts keep a slot with intervalCount == 0.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "BoundingBox.h"
#include "Ray.h"

namespace app
{

// Remaining solid along one cast, in fixed-point ticks.
// model t = originT + tick * unit
// Valid solid requires end > begin (zero/negative length intervals are never stored).
// beginNormal / endNormal are outward surface normals at the endpoints (model
// space), used by Ray-GS metal shading.
struct Interval
{
    std::int32_t begin = 0;
    std::int32_t end = 0;
    Normal3f beginNormal{0.0f, 0.0f, 0.0f};
    Normal3f endNormal{0.0f, 0.0f, 0.0f};
    std::uint8_t flags = 0; // bit0 = fromBoolean

    static constexpr std::uint8_t flagFromBoolean = 1;

    bool fromBoolean() const { return (flags & flagFromBoolean) != 0; }
    void setFromBoolean(bool on)
    {
        if (on) flags = static_cast<std::uint8_t>(flags | flagFromBoolean);
        else flags = static_cast<std::uint8_t>(flags & ~flagFromBoolean);
    }
};

struct RaySlot
{
    std::uint32_t intervalOffset = 0;
    std::uint32_t intervalCount = 0;

    bool empty() const { return intervalCount == 0; }
};

struct IntervalSpan
{
    Interval* beginPtr = nullptr;
    std::uint32_t count = 0;

    Interval* begin() const { return beginPtr; }
    Interval* end() const { return beginPtr + count; }
    const Interval* data() const { return beginPtr; }
    bool empty() const { return count == 0; }
    std::uint32_t size() const { return count; }
};

struct ConstIntervalSpan
{
    const Interval* beginPtr = nullptr;
    std::uint32_t count = 0;

    const Interval* begin() const { return beginPtr; }
    const Interval* end() const { return beginPtr + count; }
    const Interval* data() const { return beginPtr; }
    bool empty() const { return count == 0; }
    std::uint32_t size() const { return count; }
};

class IntervalPool
{
public:
    IntervalSpan span(const RaySlot& slot)
    {
        if (slot.intervalCount == 0) return {};
        return IntervalSpan{data.data() + slot.intervalOffset, slot.intervalCount};
    }

    ConstIntervalSpan span(const RaySlot& slot) const
    {
        if (slot.intervalCount == 0) return {};
        return ConstIntervalSpan{data.data() + slot.intervalOffset, slot.intervalCount};
    }

    void append(RaySlot& slot, const Interval* begin, std::uint32_t count)
    {
        slot.intervalOffset = static_cast<std::uint32_t>(data.size());
        slot.intervalCount = count;
        data.insert(data.end(), begin, begin + count);
    }

    void append(RaySlot& slot, const std::vector<Interval>& intervals)
    {
        append(slot, intervals.data(), static_cast<std::uint32_t>(intervals.size()));
    }

    bool tryReplaceInPlace(RaySlot& slot, const Interval* begin, std::uint32_t count)
    {
        if (count > slot.intervalCount) return false;
        for (std::uint32_t i = 0; i < count; ++i)
            data[slot.intervalOffset + i] = begin[i];
        slot.intervalCount = count;
        return true;
    }

    void clear() { data.clear(); }

    std::vector<Interval> data;
};

// Dense sampling grid for one cast axis. Lateral indices (iu, iv) address slots.
class RayGrid
{
public:
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    float spacingU = 1.0f;
    float spacingV = 1.0f;
    float originU = 0.0f;
    float originV = 0.0f;

    // Along-axis fixed-point map: t = originT + tick * unit
    float originT = 0.0f;
    float unit = 1.0f;

    std::size_t axis = 0; // 0=x, 1=y, 2=z

    std::vector<RaySlot> cells;
    IntervalPool pool;

    bool empty() const { return width == 0 || height == 0 || cells.empty(); }

    RaySlot& at(std::uint32_t iu, std::uint32_t iv) { return cells[iv * width + iu]; }
    const RaySlot& at(std::uint32_t iu, std::uint32_t iv) const { return cells[iv * width + iu]; }

    double sampleU(std::uint32_t iu) const
    {
        return static_cast<double>(originU) + static_cast<double>(spacingU) * static_cast<double>(iu);
    }

    double sampleV(std::uint32_t iv) const
    {
        return static_cast<double>(originV) + static_cast<double>(spacingV) * static_cast<double>(iv);
    }

    std::int32_t toTick(double t) const
    {
        if (!(unit > 0.0f)) return 0;
        const double ticks = (t - static_cast<double>(originT)) / static_cast<double>(unit);
        if (ticks >= 0.0) return static_cast<std::int32_t>(ticks + 0.5);
        return static_cast<std::int32_t>(ticks - 0.5);
    }

    double fromTick(std::int32_t tick) const
    {
        return static_cast<double>(originT) + static_cast<double>(tick) * static_cast<double>(unit);
    }

    // Inclusive index clamp into [0, width-1] / [0, height-1].
    std::uint32_t indexU(double u) const
    {
        if (width == 0 || !(spacingU > 0.0f)) return 0;
        const double raw = (u - static_cast<double>(originU)) / static_cast<double>(spacingU);
        long long i = static_cast<long long>(std::floor(raw));
        if (i < 0) i = 0;
        if (i >= static_cast<long long>(width)) i = static_cast<long long>(width) - 1;
        return static_cast<std::uint32_t>(i);
    }

    std::uint32_t indexV(double v) const
    {
        if (height == 0 || !(spacingV > 0.0f)) return 0;
        const double raw = (v - static_cast<double>(originV)) / static_cast<double>(spacingV);
        long long i = static_cast<long long>(std::floor(raw));
        if (i < 0) i = 0;
        if (i >= static_cast<long long>(height)) i = static_cast<long long>(height) - 1;
        return static_cast<std::uint32_t>(i);
    }

    std::size_t intervalCount() const
    {
        std::size_t total = 0;
        for (const RaySlot& slot : cells) total += slot.intervalCount;
        return total;
    }

    std::size_t intervalCountAtStride(int stride) const
    {
        if (stride <= 1) return intervalCount();
        std::size_t total = 0;
        for (std::uint32_t iv = 0; iv < height; ++iv)
        {
            if (static_cast<int>(iv) % stride != 0) continue;
            for (std::uint32_t iu = 0; iu < width; ++iu)
            {
                if (static_cast<int>(iu) % stride != 0) continue;
                total += at(iu, iv).intervalCount;
            }
        }
        return total;
    }
};

} // namespace app
