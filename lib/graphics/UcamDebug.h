// Debug UI gate. Extra view-mode and resolution controls stay hidden unless
// the process has UCAM_DEBUG set to kUcamDebugKey.
#pragma once

#include <cstdlib>

namespace app
{

inline constexpr char kUcamDebugEnv[] = "UCAM_DEBUG";
inline constexpr int kUcamDebugKey = 8796307;

inline bool ucamDebugEnabled()
{
    const char* value = std::getenv(kUcamDebugEnv);
    if (!value) return false;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed == kUcamDebugKey;
}

} // namespace app
