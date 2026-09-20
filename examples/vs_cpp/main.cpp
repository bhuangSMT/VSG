// Visual Studio C++ boolean sample for the UCAM SDK.
// Include <ucam/ucam.h> only. Do not include RenderManager.h or VSG.

#include <cstdint>
#include <cstdio>
#include <vector>

#include <ucam/ucam.h>

static bool fail(const char* step, UCAM::status st)
{
    if (st == UCAM::OK) return false;
    std::fprintf(stderr, "%s failed (%d): %s\n", step, static_cast<int>(st),
                 UCAM::last_error());
    return true;
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

int main()
{
    std::printf("UCAM ABI %d\n", static_cast<int>(UCAM::abi_version()));
    if (UCAM::abi_version() != UCAM::ABI_VERSION) return 1;

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
        UCAM::Boolean::stock_destroy(stock);
        UCAM::Boolean::session_destroy(session);
        return 1;
    }
    std::printf("cast rays: %llu\n",
                static_cast<unsigned long long>(UCAM::Boolean::stock_ray_count(stock)));

    UCAM::Boolean::sweep* sweep = nullptr;
    if (fail("sweep_begin",
             UCAM::Boolean::sweep_begin(session, UCAM::Boolean::TOOL_FLAT_NOSE, 0.08, 0.22,
                                        &sweep)))
    {
        UCAM::Boolean::stock_destroy(stock);
        UCAM::Boolean::session_destroy(session);
        return 1;
    }
    UCAM::Boolean::sweep_add_pose(sweep, -0.25, 0.0, 0.15, 0.0, 0.0, 1.0);
    UCAM::Boolean::sweep_add_pose(sweep, 0.25, 0.05, 0.10, 0.0, 0.0, 1.0);
    if (fail("sweep_end", UCAM::Boolean::sweep_end(sweep)))
    {
        UCAM::Boolean::sweep_destroy(sweep);
        UCAM::Boolean::stock_destroy(stock);
        UCAM::Boolean::session_destroy(session);
        return 1;
    }

    if (fail("apply", UCAM::Boolean::apply(stock, sweep, UCAM::Boolean::OP_SUBTRACTION)))
    {
        UCAM::Boolean::sweep_destroy(sweep);
        UCAM::Boolean::stock_destroy(stock);
        UCAM::Boolean::session_destroy(session);
        return 1;
    }
    std::printf("after subtract rays: %llu  dirty cells: %llu\n",
                static_cast<unsigned long long>(UCAM::Boolean::stock_ray_count(stock)),
                static_cast<unsigned long long>(UCAM::Boolean::stock_dirty_cell_count(stock)));

    UCAM::Boolean::sweep_destroy(sweep);
    UCAM::Boolean::stock_destroy(stock);
    UCAM::Boolean::session_destroy(session);
    return 0;
}
