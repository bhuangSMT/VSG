// Axis-aligned bounding box in model space.
#pragma once

#include <cstddef>

#include "Ray.h"

namespace app
{

class BRep;

class BoundingBox
{
public:
    // A default constructed box is empty: valid() is false until a point is
    // added, so that min/max are never mistaken for a real extent.
    BoundingBox() = default;
    BoundingBox(const Point3d& minCorner, const Point3d& maxCorner);

    static BoundingBox fromBRep(const BRep& brep);

    // Grow the box to include p.
    void expand(const Point3d& p);

    bool valid() const { return _valid; }

    bool contains(const Point3d& p) const
    {
        if (!_valid) return false;
        return p[0] >= _min[0] && p[0] <= _max[0] && p[1] >= _min[1] && p[1] <= _max[1] &&
               p[2] >= _min[2] && p[2] <= _max[2];
    }

    const Point3d& min() const { return _min; }
    const Point3d& max() const { return _max; }

    // axis is 0 for x, 1 for y, 2 for z.
    double extent(std::size_t axis) const;

    // Length of the box diagonal; 0 for an empty box.
    double diagonal() const;

    Point3d centre() const;

private:
    Point3d _min{0.0, 0.0, 0.0};
    Point3d _max{0.0, 0.0, 0.0};
    bool _valid = false;
};

} // namespace app
