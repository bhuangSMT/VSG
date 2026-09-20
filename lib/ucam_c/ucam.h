// UCAM C++ SDK — the only header a third-party project should include.
//
// Boolean and Graphics wrap the in-tree engines. No VSG, Qt, or Eigen types
// appear here. C is not supported.
#pragma once

#ifndef __cplusplus
#error "UCAM requires a C++17 compiler. Include <ucam/ucam.h> from C++."
#endif

#include <cstdint>

#define UCAM_ABI_VERSION 2

#if defined(_WIN32)
#if defined(UCAM_EXPORTS)
#define UCAM_API __declspec(dllexport)
#else
#define UCAM_API __declspec(dllimport)
#endif
#else
#define UCAM_API __attribute__((visibility("default")))
#endif

namespace UCAM
{

constexpr int32_t ABI_VERSION = 2;

using status = int32_t;
constexpr status OK = 0;
constexpr status ERR_INVALID_ARG = 1;
constexpr status ERR_FAILED = 2;
constexpr status ERR_UNSUPPORTED = 3;

UCAM_API int32_t abi_version();
UCAM_API const char* last_error();

namespace Boolean
{

struct session;
struct stock;
struct sweep;

enum tool_type
{
    TOOL_NONE = 0,
    TOOL_BULL_NOSE = 1,
    TOOL_FLAT_NOSE = 2,
    TOOL_BALL_NOSE = 3,
    TOOL_SPHERE = 4,
    TOOL_GRINDING_WHEEL = 5
};

enum boolean_op
{
    OP_NONE = 0,
    OP_PROBE = 1,
    OP_SUBTRACTION = 2,
    OP_UNION = 3,
    OP_INSPECTION = 4
};

UCAM_API status session_create(session** out);
UCAM_API void session_destroy(session* session);

// xyz: 3 * vertex_count floats. indices: 3 * triangle_count uint32s (CCW).
UCAM_API status stock_from_triangles(session* session,
                                     const float* xyz,
                                     uint32_t vertex_count,
                                     const uint32_t* indices,
                                     uint32_t triangle_count,
                                     stock** out);
UCAM_API void stock_destroy(stock* stock);

UCAM_API status stock_cast(stock* stock, double res_x, double res_y, double res_z);
UCAM_API uint64_t stock_ray_count(const stock* stock);
UCAM_API uint64_t stock_dirty_cell_count(const stock* stock);

// Copies the original (pre-boolean) triangle soup. Pass null buffers to query
// counts. xyz is 3 * vertex_count; indices is 3 * triangle_count.
UCAM_API status stock_export_triangles(const stock* stock,
                                       float* xyz,
                                       uint32_t* vertex_count,
                                       uint32_t* indices,
                                       uint32_t* triangle_count);

UCAM_API status sweep_begin(session* session,
                            int32_t tool_type,
                            double radius,
                            double length,
                            sweep** out);
UCAM_API status sweep_add_pose(sweep* sweep,
                               double x, double y, double z,
                               double dx, double dy, double dz);
UCAM_API status sweep_end(sweep* sweep);
UCAM_API void sweep_destroy(sweep* sweep);

// Model-space identity: stock and tool poses share the same coordinates.
UCAM_API status apply(stock* stock, sweep* sweep, int32_t op);

} // namespace Boolean

namespace Graphics
{

struct window;
struct view;

// QApplication + default Vulkan traits. Call before window_create.
UCAM_API status init(int* argc, char** argv);
UCAM_API void shutdown();

UCAM_API status window_create(int32_t width, int32_t height, const char* title,
                              window** out);
UCAM_API void window_destroy(window* window);

// Bind a live Disk view to the window (owns RenderManager inside the dylib).
UCAM_API status create(window* window, view** out);
UCAM_API void destroy(view* view);

UCAM_API status set_stock(view* view, Boolean::stock* stock);
UCAM_API status set_tool(view* view, int32_t tool_type, double radius, double length);
UCAM_API status commit_pose(view* view, double x, double y, double z,
                            double dx, double dy, double dz);
// disk != 0 selects Disk; cut_mesh != 0 selects cut-mesh overlay.
UCAM_API status set_flags(view* view, int32_t disk, int32_t cut_mesh);

// Process Qt/Vulkan events and draw one frame. Returns OK while the window is
// open, ERR_FAILED after the user closes it. Does not block.
UCAM_API status poll(window* window);

// Show and block until close (static preview).
UCAM_API status run(window* window);

// CPU fallback (no window): tight RGBA8, row pitch = width * 4.
UCAM_API status create_offscreen(int32_t width, int32_t height, view** out);
UCAM_API status frame(view* view, uint8_t* rgba8, int32_t width, int32_t height);

} // namespace Graphics

} // namespace UCAM
