// BooleanOp - how the swept volume is combined with the RayModel.
#pragma once

namespace app
{

enum class BooleanOp
{
    None,
    Subtraction,
    Union,
    // Preview only: restore the cached original RayModel each mouse move,
    // subtract the cutter at the current pose, and do not accumulate cuts.
    Inspection
};

} // namespace app
