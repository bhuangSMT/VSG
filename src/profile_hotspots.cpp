// profile_hotspots - headless timing of cast / sweep / boolean / splat paths.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>

#include "BRep.h"
#include "BoundingBox.h"
#include "BooleanOp.h"
#include "GaussianSplatCache.h"
#include "RayBoolean.h"
#include "RayModel.h"
#include "SweptVolume.h"
#include "ToolType.h"
#include "TriangleMesh.h"

namespace
{

using Clock = std::chrono::steady_clock;

struct Timed
{
    const char* name = "";
    double ms = 0.0;
    std::string detail;
};

app::BRep createCubeBRep()
{
    const vsg::vec3 corners[8] = {
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};

    const int triangles[12][3] = {
        {0, 3, 2}, {0, 2, 1},
        {4, 5, 6}, {4, 6, 7},
        {0, 1, 5}, {0, 5, 4},
        {1, 2, 6}, {1, 6, 5},
        {2, 3, 7}, {2, 7, 6},
        {3, 0, 4}, {3, 4, 7}};

    app::TriangleMesh mesh;
    mesh.name = "cube";
    for (const auto& tri : triangles)
    {
        app::MeshTriangle t;
        t.v0 = corners[tri[0]];
        t.v1 = corners[tri[1]];
        t.v2 = corners[tri[2]];
        const vsg::vec3 e1 = t.v1 - t.v0;
        const vsg::vec3 e2 = t.v2 - t.v0;
        t.normal = vsg::normalize(vsg::cross(e1, e2));
        mesh.triangles.push_back(t);
    }
    return app::BRep::fromTriangles(mesh);
}

double elapsedMs(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

app::SweptVolume makeSweep(app::ToolType type, double radius, double length)
{
    app::SweptVolume sweep;
    const app::ToolPose a{vsg::dvec3(-0.25, 0.0, 0.15), vsg::dvec3(0.0, 0.0, 1.0)};
    const app::ToolPose b{vsg::dvec3(0.25, 0.05, 0.10), vsg::dvec3(0.0, 0.0, 1.0)};
    sweep.appendSegment(type, static_cast<float>(radius), static_cast<float>(length), a, b, 12);
    return sweep;
}

std::array<float, 3> radiiFor(const app::RayModel& model, int stride)
{
    const app::Point3d& r = model.resolution();
    auto one = [&](std::size_t axis) {
        const double du = r[(axis + 1) % 3];
        const double dv = r[(axis + 2) % 3];
        double radius = 0.5 * std::sqrt(du * du + dv * dv) * 2.0;
        return static_cast<float>(radius * static_cast<double>(stride));
    };
    return {one(0), one(1), one(2)};
}

} // namespace

int main(int argc, char** argv)
{
    int repeats = 5;
    double resolution = 0.02;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if ((arg == "--repeats" || arg == "-n") && i + 1 < argc)
            repeats = std::max(1, std::atoi(argv[++i]));
        else if ((arg == "--resolution" || arg == "-r") && i + 1 < argc)
            resolution = std::atof(argv[++i]);
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: profile_hotspots [--repeats N] [--resolution R]\n";
            return 0;
        }
    }

    const app::BRep brep = createCubeBRep();
    const app::Point3d res{resolution, resolution, resolution};
    const double toolR = 0.08;
    const double toolL = toolR * 2.8;

    std::vector<Timed> rows;
    rows.reserve(32);

    // --- Cast ---
    {
        double total = 0.0;
        std::size_t intervals = 0;
        for (int i = 0; i < repeats; ++i)
        {
            const auto t0 = Clock::now();
            app::RayModel model = app::RayModel::fromBRep(brep, res);
            total += elapsedMs(t0);
            intervals = model.rayCount();
        }
        rows.push_back({"fromBRep (cast)", total / repeats,
                        std::to_string(intervals) + " intervals @ res=" +
                            std::to_string(resolution)});
    }

    app::RayModel stock = app::RayModel::fromBRep(brep, res);
    const int stride = stock.strideForRayBudget(750000);
    const auto splatRadii = radiiFor(stock, stride);
    const app::SplatStyle style{};

    // --- Sweep mesh build (per tool) ---
    for (app::ToolType type : {app::ToolType::Sphere, app::ToolType::BallNose,
                               app::ToolType::FlatNose, app::ToolType::BullNose})
    {
        const char* name =
            type == app::ToolType::Sphere     ? "appendSegment Sphere"
            : type == app::ToolType::BallNose ? "appendSegment BallNose"
            : type == app::ToolType::FlatNose ? "appendSegment FlatNose"
                                              : "appendSegment BullNose";
        double total = 0.0;
        std::size_t tris = 0;
        for (int i = 0; i < repeats; ++i)
        {
            const auto t0 = Clock::now();
            app::SweptVolume sweep = makeSweep(type, toolR, toolL);
            total += elapsedMs(t0);
            tris = sweep.mesh().triangles.size();
        }
        rows.push_back({name, total / repeats, std::to_string(tris) + " tris"});
    }

    app::SweptVolume sphereSweep = makeSweep(app::ToolType::Sphere, toolR, toolL);
    app::SweptVolume ballSweep = makeSweep(app::ToolType::BallNose, toolR, toolL);
    app::SweptVolume flatSweep = makeSweep(app::ToolType::FlatNose, toolR, toolL);
    const vsg::dmat4 identity{};

    // --- Boolean ---
    // Copy+mutate (withBoolean) — first-cut / fork cost.
    auto timeBooleanCopy = [&](const char* label, app::SweptVolume& sweep) {
        double total = 0.0;
        std::size_t outIntervals = 0;
        for (int i = 0; i < repeats; ++i)
        {
            const auto t0 = Clock::now();
            app::RayModel cut =
                stock.withBoolean(sweep, app::BooleanOp::Subtraction, identity);
            total += elapsedMs(t0);
            outIntervals = cut.rayCount();
        }
        rows.push_back({label, total / repeats,
                        std::to_string(outIntervals) + " intervals after cut"});
    };
    timeBooleanCopy("boolean copy Sphere", sphereSweep);
    timeBooleanCopy("boolean copy BallNose", ballSweep);
    timeBooleanCopy("boolean copy FlatNose", flatSweep);

    // In-place mutate (interactive steady-state after first fork).
    {
        double total = 0.0;
        std::size_t outIntervals = 0;
        for (int i = 0; i < repeats; ++i)
        {
            app::RayModel working = stock.clone();
            const auto t0 = Clock::now();
            working.booleanInPlace(sphereSweep, app::BooleanOp::Subtraction, identity);
            total += elapsedMs(t0);
            outIntervals = working.rayCount();
        }
        rows.push_back({"boolean inPlace Sphere", total / repeats,
                        std::to_string(outIntervals) + " intervals after cut"});
    }

    // Cumulative boolean (10 successive in-place cuts) — interactive path.
    {
        double total = 0.0;
        app::RayModel working = app::RayModel::fromBRep(brep, res);
        const int steps = 10;
        for (int s = 0; s < steps; ++s)
        {
            app::SweptVolume step;
            const double x0 = -0.3 + 0.05 * s;
            const app::ToolPose a{vsg::dvec3(x0, 0.0, 0.12), vsg::dvec3(0.0, 0.0, 1.0)};
            const app::ToolPose b{vsg::dvec3(x0 + 0.05, 0.0, 0.12), vsg::dvec3(0.0, 0.0, 1.0)};
            step.appendSegment(app::ToolType::Sphere, static_cast<float>(toolR),
                               static_cast<float>(toolL), a, b, 12);
            const auto t0 = Clock::now();
            working.booleanInPlace(step, app::BooleanOp::Subtraction, identity);
            total += elapsedMs(t0);
        }
        rows.push_back({"boolean cumulative x10", total,
                        std::to_string(working.rayCount()) + " intervals (sum of 10 cuts)"});
        rows.push_back({"boolean cumulative /cut", total / steps, "avg of 10 successive cuts"});
    }

    // --- Splat cache ---
    {
        app::GaussianSplatCache cache;
        double coldTotal = 0.0;
        for (int i = 0; i < repeats; ++i)
        {
            cache.release();
            const auto t0 = Clock::now();
            cache.rebuild(stock, stride, splatRadii, style);
            coldTotal += elapsedMs(t0);
        }
        rows.push_back({"splatCache.rebuild cold", coldTotal / repeats,
                        "stride=" + std::to_string(stride) + " live=" +
                            std::to_string(cache.liveEndpoints()) +
                            " cap=" + std::to_string(cache.capacity())});

        double warmTotal = 0.0;
        for (int i = 0; i < repeats; ++i)
        {
            cache.clear(); // soft: keep buffers
            const auto t0 = Clock::now();
            cache.rebuild(stock, stride, splatRadii, style);
            warmTotal += elapsedMs(t0);
        }
        rows.push_back({"splatCache.rebuild warm", warmTotal / repeats,
                        "reuse buffers/pipelines"});
    }

    {
        app::GaussianSplatCache cache;
        cache.rebuild(stock, stride, splatRadii, style);
        app::RayModel cut =
            stock.withBoolean(sphereSweep, app::BooleanOp::Subtraction, identity);
        const app::BoundingBox dirty =
            app::modelAabbFromWorld(sphereSweep.bvh().bounds(), vsg::dmat4{});

        double total = 0.0;
        for (int i = 0; i < repeats; ++i)
        {
            // Refresh region from already-cut model (steady-state update cost).
            const auto t0 = Clock::now();
            const bool ok = cache.updateRegion(cut, dirty, stride, splatRadii, style);
            total += elapsedMs(t0);
            if (!ok)
            {
                rows.push_back({"splatCache.updateRegion", total / (i + 1), "FAILED (fell back)"});
                break;
            }
            if (i + 1 == repeats)
                rows.push_back({"splatCache.updateRegion", total / repeats, "AABB patch"});
        }
    }

    // Print report sorted by time descending for the single-op averages.
    std::cout << "\n=== Hotspot profile (cube 1x1x1, repeats=" << repeats
              << ", res=" << resolution << ") ===\n";
    std::cout << std::left << std::setw(28) << "operation" << std::right << std::setw(12)
              << "ms" << "  " << "detail\n";
    std::cout << std::string(70, '-') << '\n';

    std::vector<Timed> sortable = rows;
    std::sort(sortable.begin(), sortable.end(),
              [](const Timed& a, const Timed& b) { return a.ms > b.ms; });

    for (const Timed& row : sortable)
    {
        std::cout << std::left << std::setw(28) << row.name << std::right << std::setw(12)
                  << std::fixed << std::setprecision(2) << row.ms << "  " << row.detail
                  << '\n';
    }

    const Timed* worst = nullptr;
    for (const Timed& row : sortable)
    {
        // Prefer single-op averages over the cumulative sum row.
        if (std::string(row.name).find("cumulative x10") != std::string::npos) continue;
        worst = &row;
        break;
    }

    std::cout << std::string(70, '-') << '\n';
    if (worst)
    {
        std::cout << "Bottleneck (slowest single-op avg): " << worst->name << " @ "
                  << std::fixed << std::setprecision(2) << worst->ms << " ms\n";
    }
    std::cout << '\n';
    return 0;
}
