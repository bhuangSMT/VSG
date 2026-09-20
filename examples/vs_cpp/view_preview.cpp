// Visual Studio C++ graphics sample for the UCAM SDK.
// Include <ucam/ucam.h> only. Do not include RenderManager.h, Qt, or VSG.
//
// Boolean first, then Graphics::init / window_create / create, then a pose-tick
// loop (apply + commit_pose + poll). Copy the SDK bin/ folder next to the exe.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <ucam/ucam.h>

struct Pose
{
    double x, y, z, dx, dy, dz;
};

static bool fail(const char* step, UCAM::status st)
{
    if (st == UCAM::OK) return false;
    std::fprintf(stderr, "%s failed (%d): %s\n", step, static_cast<int>(st),
                 UCAM::last_error());
    return true;
}

static void cleanup(UCAM::Graphics::view* view, UCAM::Graphics::window* window,
                    UCAM::Boolean::stock* stock, UCAM::Boolean::session* session)
{
    UCAM::Graphics::destroy(view);
    UCAM::Graphics::window_destroy(window);
    UCAM::Graphics::shutdown();
    UCAM::Boolean::stock_destroy(stock);
    UCAM::Boolean::session_destroy(session);
}

static void makeUnitBox(std::vector<float>& xyz, std::vector<uint32_t>& indices)
{
    const float c[8][3] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},  {0.5f, 0.5f, 0.5f},  {-0.5f, 0.5f, 0.5f},
    };
    xyz.resize(8 * 3);
    for (int i = 0; i < 8; ++i)
    {
        xyz[static_cast<size_t>(i) * 3 + 0] = c[i][0];
        xyz[static_cast<size_t>(i) * 3 + 1] = c[i][1];
        xyz[static_cast<size_t>(i) * 3 + 2] = c[i][2];
    }
    const uint32_t tris[] = {
        0, 3, 2, 0, 2, 1, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
        1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    };
    indices.assign(tris, tris + 36);
}

int main(int argc, char** argv)
{
    if (UCAM::abi_version() != UCAM::ABI_VERSION)
    {
        std::fprintf(stderr, "ABI mismatch: got %d expected %d\n",
                     static_cast<int>(UCAM::abi_version()),
                     static_cast<int>(UCAM::ABI_VERSION));
        return 1;
    }

    UCAM::Boolean::session* session = nullptr;
    if (fail("session_create", UCAM::Boolean::session_create(&session))) return 1;

    std::vector<float> xyz;
    std::vector<uint32_t> indices;
    makeUnitBox(xyz, indices);

    UCAM::Boolean::stock* stock = nullptr;
    if (fail("stock_from_triangles",
             UCAM::Boolean::stock_from_triangles(session, xyz.data(), 8, indices.data(), 12,
                                                 &stock)))
    {
        UCAM::Boolean::session_destroy(session);
        return 1;
    }

    if (fail("stock_cast", UCAM::Boolean::stock_cast(stock, 0.05, 0.05, 0.05)))
    {
        cleanup(nullptr, nullptr, stock, session);
        return 1;
    }

    constexpr double kRadius = 0.08;
    constexpr double kLength = 0.22;
    const Pose poses[] = {
        {-0.25, 0.0, 0.18, 0.0, 0.0, 1.0},
        {-0.08, 0.0, 0.14, 0.0, 0.0, 1.0},
        {0.08, 0.02, 0.12, 0.0, 0.0, 1.0},
        {0.25, 0.05, 0.10, 0.0, 0.0, 1.0},
    };
    const size_t nposes = sizeof(poses) / sizeof(poses[0]);

    if (fail("Graphics::init", UCAM::Graphics::init(&argc, argv)))
    {
        cleanup(nullptr, nullptr, stock, session);
        return 1;
    }

    UCAM::Graphics::window* window = nullptr;
    if (fail("window_create",
             UCAM::Graphics::window_create(800, 600, "UCAM preview", &window)))
    {
        cleanup(nullptr, nullptr, stock, session);
        return 1;
    }

    UCAM::Graphics::view* view = nullptr;
    if (fail("create", UCAM::Graphics::create(window, &view)))
    {
        cleanup(nullptr, window, stock, session);
        return 1;
    }

    if (fail("set_tool", UCAM::Graphics::set_tool(view, UCAM::Boolean::TOOL_FLAT_NOSE,
                                                  kRadius, kLength)))
    {
        cleanup(view, window, stock, session);
        return 1;
    }
    UCAM::Graphics::set_flags(view, 1, 1);

    for (size_t i = 1; i < nposes; ++i)
    {
        UCAM::Boolean::sweep* sweep = nullptr;
        if (fail("sweep_begin",
                 UCAM::Boolean::sweep_begin(session, UCAM::Boolean::TOOL_FLAT_NOSE, kRadius,
                                            kLength, &sweep)))
        {
            cleanup(view, window, stock, session);
            return 1;
        }
        UCAM::Boolean::sweep_add_pose(sweep, poses[i - 1].x, poses[i - 1].y, poses[i - 1].z,
                                      poses[i - 1].dx, poses[i - 1].dy, poses[i - 1].dz);
        UCAM::Boolean::sweep_add_pose(sweep, poses[i].x, poses[i].y, poses[i].z, poses[i].dx,
                                      poses[i].dy, poses[i].dz);
        if (fail("sweep_end", UCAM::Boolean::sweep_end(sweep)) ||
            fail("apply", UCAM::Boolean::apply(stock, sweep, UCAM::Boolean::OP_SUBTRACTION)))
        {
            UCAM::Boolean::sweep_destroy(sweep);
            cleanup(view, window, stock, session);
            return 1;
        }
        UCAM::Boolean::sweep_destroy(sweep);

        if (fail("set_stock", UCAM::Graphics::set_stock(view, stock)) ||
            fail("commit_pose",
                 UCAM::Graphics::commit_pose(view, poses[i].x, poses[i].y, poses[i].z,
                                             poses[i].dx, poses[i].dy, poses[i].dz)))
        {
            cleanup(view, window, stock, session);
            return 1;
        }
        if (UCAM::Graphics::poll(window) != UCAM::OK)
        {
            cleanup(view, window, stock, session);
            return 0;
        }
    }

    std::puts("playback done — close the window to exit");
    while (UCAM::Graphics::poll(window) == UCAM::OK)
    {
    }

    cleanup(view, window, stock, session);
    return 0;
}
