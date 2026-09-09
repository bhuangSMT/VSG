#include "Parameter.h"

namespace app
{

Parameter& Parameter::instance()
{
    // A function-local static: built on the first call and torn down at exit.
    // C++11 onwards guarantees the construction is race free, so callers do not
    // have to coordinate.
    static Parameter store;
    return store;
}

} // namespace app
