# UCAM C++ SDK

This package exposes the UCAM **Boolean** and **Graphics** engines through a
C++ API (`namespace UCAM`). Your project includes one header and links `ucam`.
You do **not** add VSG, Qt, or Eigen include paths.

Requires **C++17**. There is no C ABI.

Two platform zips (same folders, native binaries):

- `ucam-sdk-windows-x64.zip`
- `ucam-sdk-macos-arm64.zip`

## Layout

```
ucam-sdk-<os>-<arch>/
  include/ucam/ucam.h     # UCAM::Boolean, UCAM::Graphics
  lib/                    # link against these
  bin/                    # copy next to your executable (all of it)
  examples/vs_cpp/
  README.md
```

**`include/`** — `<ucam/ucam.h>` only.

**`lib/`** — import / link libraries. Windows: `ucam.lib` plus Qt, VSG, vsgQt,
Vulkan, TBB `.lib` files. macOS: `libucam.dylib` plus the matching dylibs.
The sample only names `ucam` (`ucam.lib` / `-lucam`).

**`bin/`** — runtime. Copy the **entire** tree next to the exe, including
`platforms/qwindows.dll` (Windows) or `plugins/platforms/libqcocoa.dylib`
(macOS). Missing the platform plugin looks like a blank window.

Do **not** include `RenderManager.h` or link `ucam_graphics` as a public API.

## Visual Studio 2022 (x64)

1. Unzip to e.g. `C:\libs\ucam-sdk`.
2. New Console App, platform **x64**, C++17.
3. Additional Include Directories: `$(UCAM_SDK)\include`
4. Additional Library Directories: `$(UCAM_SDK)\lib`
5. Additional Dependencies: `ucam.lib`
6. Post-Build: `xcopy /E /Y "$(UCAM_SDK)\bin\*" "$(OutDir)"`

Optional user macro `UCAM_SDK` = `C:\libs\ucam-sdk`.

## macOS (Xcode / CMake)

```
-I /path/to/ucam-sdk/include
-L /path/to/ucam-sdk/lib -lucam
```

Copy `bin/` next to the executable (or set `DYLD_LIBRARY_PATH` to `bin`).

CMake:

```cmake
find_package(ucam REQUIRED)
target_link_libraries(their_app PRIVATE ucam::ucam)
```

```cpp
#include <ucam/ucam.h>
```

## Coordinates

Positions and tool poses are in **model units** (the same space as your
triangle vertices). The SDK does not apply this application's unit-box fit.

## Errors and ABI

Every call returns `UCAM::status`. `UCAM::OK` is 0. On failure read
`UCAM::last_error()`. `UCAM::abi_version()` must equal `UCAM::ABI_VERSION` (2).

## Boolean example

`examples/vs_cpp/main.cpp`. Unit box, cast, flat-nose sweep,
`UCAM::Boolean::apply(..., OP_SUBTRACTION)`.

## Graphics (window + animation)

`examples/vs_cpp/view_preview.cpp`. Graphics owns Qt/VSG/RenderManager inside
`libucam`. You still include only `<ucam/ucam.h>`.

1. Boolean / cast a stock.
2. `UCAM::Graphics::init(&argc, argv)` — `QApplication` + Vulkan traits.
3. `window_create(800, 600, "UCAM preview")`
4. `create(window)` — bind a view (camera, trackball, RenderManager).
5. `set_tool` / `set_flags(1, 1)` — Disk + cut-mesh.
6. Each animation step: `sweep_*` + `apply`, then `set_stock`, `commit_pose`,
   `poll`. `poll` draws one frame and returns `ERR_FAILED` when the window is
   closed.
7. `destroy` / `window_destroy` / `shutdown`.

```cpp
UCAM::Graphics::init(&argc, argv);
UCAM::Graphics::window* window = nullptr;
UCAM::Graphics::window_create(800, 600, "UCAM preview", &window);
UCAM::Graphics::view* view = nullptr;
UCAM::Graphics::create(window, &view);
UCAM::Graphics::set_tool(view, UCAM::Boolean::TOOL_FLAT_NOSE, 0.08, 0.22);
UCAM::Graphics::set_flags(view, 1, 1);
UCAM::Graphics::set_stock(view, stock);
UCAM::Graphics::commit_pose(view, 0.25, 0.05, 0.10, 0.0, 0.0, 1.0);
while (UCAM::Graphics::poll(window) == UCAM::OK) {}
```

`create_offscreen` + `frame` is a CPU RGBA8 fallback (no window). Cut orange
`(242, 89, 26)`, stock gray. Top-down XY of the AABB.

If your app already uses Qt, use the **same** Qt major as this `bin/` — do not
mix two Qt builds. Prefer not linking Qt yourself.

## Building the SDK from this repository

macOS:

```
./scripts/package_sdk.sh
```

writes `dist/ucam-sdk-macos-arm64.zip`.

Windows (from a Windows build tree):

```
powershell -File scripts/package_sdk.ps1
```

writes `dist/ucam-sdk-windows-x64.zip`.

`vsg_qt_cube` continues to link the internal C++ libraries. Third parties use
only `ucam::ucam`.

## Internal (this repository only)

`vsg_qt_cube` constructs `app::RenderManager` with a live `vsgQt::Viewer`.
Integrators should not copy those headers.

```cpp
auto renderManager = std::make_shared<app::RenderManager>(viewer, vsg_scene, options);
renderManager->showBRep(startupBRep);
renderManager->setRayModel(app::RayModel::fromBRep(*brep, resolution));
renderManager->setViewMode(store.viewMode());
renderManager->setToolType(type);
renderManager->setToolPose(position, direction, /*commitSweep*/ true);
```
