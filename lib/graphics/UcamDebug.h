// Debug UI gate. Extra view-mode and resolution controls stay hidden unless
// the process has UCAM_DEBUG set to kUcamDebugKey.
#pragma once

#include <cstdlib>
#include <cstring>

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

// Splat shader diagnostic override, read once when the pipelines are built.
// Empty (the default) leaves the shaders byte-identical to the release path.
//
// flat     albedo only, no lighting: separates a normal problem from a
//          coverage / depth problem
// normal   shaded normal as RGB: axis-coloured patches mean the degenerate
//          fallback is firing, speckle means the recorded field is noisy
// diffuse  the diffuse term alone, without albedo or specular
// edge     edgeStrength in red, "narrowing active" in green
// depth    banded depth ramp: shows when a mark belongs to a farther surface
// shrink   narrow the crease-facing side of each disc by edgeStrength
inline constexpr char kUcamSplatDebugEnv[] = "UCAM_SPLAT_DEBUG";

inline const char* ucamSplatDebugMode()
{
    const char* value = std::getenv(kUcamSplatDebugEnv);
    return value ? value : "";
}

// GLSL define for the requested mode, or nullptr when unset / unrecognised.
inline const char* ucamSplatDebugDefine()
{
    struct Mode
    {
        const char* name;
        const char* define;
    };
    static constexpr Mode modes[] = {
        {"flat", "SPLAT_DEBUG_FLAT"},   {"normal", "SPLAT_DEBUG_NORMAL"},
        {"diffuse", "SPLAT_DEBUG_DIFFUSE"}, {"edge", "SPLAT_DEBUG_EDGE"},
        {"depth", "SPLAT_DEBUG_DEPTH"}, {"shrink", "SPLAT_DEBUG_SHRINK"}};

    const char* mode = ucamSplatDebugMode();
    if (*mode == '\0') return nullptr;
    for (const Mode& m : modes)
    {
        if (std::strcmp(mode, m.name) == 0) return m.define;
    }
    return nullptr;
}

} // namespace app
