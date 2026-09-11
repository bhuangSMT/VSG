// BooleanOp - how the swept volume is combined with the RayModel.
#pragma once

namespace app
{

enum class BooleanOp
{
    None,   // no cut, do not record Interactive poses
    Probe,  // no cut, record Interactive poses
    Subtraction,
    Union,
    // Preview only: restore the cached original RayModel each mouse move,
    // subtract the cutter at the current pose, and do not accumulate cuts.
    Inspection
};

inline bool appliesBoolean(BooleanOp op)
{
    return op == BooleanOp::Subtraction || op == BooleanOp::Union ||
           op == BooleanOp::Inspection;
}

} // namespace app
