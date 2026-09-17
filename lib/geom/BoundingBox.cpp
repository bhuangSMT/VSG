#include "BoundingBox.h"

#include <algorithm>
#include <cmath>

#include "BRep.h"

namespace app
{

BoundingBox::BoundingBox(const Point3d& minCorner, const Point3d& maxCorner) :
    _min(minCorner),
    _max(maxCorner),
    _valid(true)
{
}

BoundingBox BoundingBox::fromBRep(const BRep& brep)
{
    BoundingBox box;
    for (const auto& v : brep.vertices())
    {
        box.expand(Point3d{static_cast<double>(v.x),
                           static_cast<double>(v.y),
                           static_cast<double>(v.z)});
    }
    return box;
}

void BoundingBox::expand(const Point3d& p)
{
    if (!_valid)
    {
        _min = p;
        _max = p;
        _valid = true;
        return;
    }

    for (std::size_t i = 0; i < 3; ++i)
    {
        _min[i] = std::min(_min[i], p[i]);
        _max[i] = std::max(_max[i], p[i]);
    }
}

void BoundingBox::expand(const BoundingBox& other)
{
    if (!other._valid) return;
    expand(other._min);
    expand(other._max);
}

BoundingBox BoundingBox::intersection(const BoundingBox& other) const
{
    BoundingBox out;
    if (!_valid || !other._valid) return out;

    Point3d lo{};
    Point3d hi{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        lo[i] = std::max(_min[i], other._min[i]);
        hi[i] = std::min(_max[i], other._max[i]);
        if (lo[i] > hi[i]) return out;
    }
    return BoundingBox(lo, hi);
}

double BoundingBox::extent(std::size_t axis) const
{
    if (!_valid || axis > 2) return 0.0;
    return _max[axis] - _min[axis];
}

double BoundingBox::diagonal() const
{
    if (!_valid) return 0.0;

    double sum = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        const double d = _max[i] - _min[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

Point3d BoundingBox::centre() const
{
    if (!_valid) return Point3d{0.0, 0.0, 0.0};

    return Point3d{(_min[0] + _max[0]) * 0.5,
                   (_min[1] + _max[1]) * 0.5,
                   (_min[2] + _max[2]) * 0.5};
}

} // namespace app
