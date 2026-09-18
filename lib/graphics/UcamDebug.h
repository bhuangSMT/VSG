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

// Set UCAM_NORMAL_STATS=1 to print, after each splat rebuild, how many endpoint
// normals disagree with every neighbour, split by whether the boolean recorded
// them or the cache estimated them, and by how close to grazing the ray was.
// Off by default: the counters are atomics written from the parallel fill.
inline constexpr char kUcamNormalStatsEnv[] = "UCAM_NORMAL_STATS";

// Read once: this is queried per endpoint from every worker in the parallel
// fill, and getenv on that path is slow enough to look like a hang.
inline bool ucamNormalStatsEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv(kUcamNormalStatsEnv);
        return value && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Set UCAM_REPAIR_CUT_NORMALS=0 to skip the post-boolean neighbour-consensus
// pass. Default on. Restart the process to change it (read once).
inline constexpr char kUcamRepairCutNormalsEnv[] = "UCAM_REPAIR_CUT_NORMALS";

inline bool ucamRepairCutNormalsEnabled()
{
    static const bool enabled = [] {
        const char* value = std::getenv(kUcamRepairCutNormalsEnv);
        if (!value || *value == '\0') return true;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

} // namespace app
