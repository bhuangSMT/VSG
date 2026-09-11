# vsg_qt_cube

A minimal C++/CMake project that renders a **cube** with
[VulkanSceneGraph (VSG)](https://github.com/vsg-dev/VulkanSceneGraph) inside a
**Qt** `QMainWindow`, using the
[vsgQt](https://github.com/vsg-dev/vsgQt) integration layer.

```
UCAM/
├── CMakeLists.txt              # find_package(vsg, vsgQt, Vulkan, Qt6) + app + ucam libs
├── cmake/UcamLibrary.cmake     # shared vs static add_library helper
├── cmake/ucamConfig.cmake.in   # find_package(ucam) package file
├── lib/geom/                   # ucam_geom
│   ├── TriangleMesh.h          # the triangle soup every importer produces
│   ├── BRep.*                  # boundary rep (CSR topology) + watertight validation
│   ├── BVH.*                   # face hierarchy built with the BRep, queried when casting
│   ├── Ray.h                   # Point3d, Ray, RayChain (with its grid position)
│   ├── BoundingBox.*           # axis-aligned extent used to lay out the ray grid
│   ├── SweptVolume.*           # tool body swept along a path
│   ├── ToolGeometry.*          # mill / drill / etc. solid
│   └── ToolType.h
├── lib/boolean/                # ucam_boolean
│   ├── BooleanOp.h
│   ├── RayGrid.h
│   ├── RayHit.*
│   ├── RayModel.*              # axis-aligned rays cast through a BRep
│   └── RayBoolean.*
├── lib/graphics/               # ucam_graphics
│   ├── GaussianSplat.*         # point set -> camera-facing Gaussian splats
│   ├── GaussianSplatCache.*
│   ├── RenderManager.*         # BRep <-> VSG scene handshake (build, compile, attach)
│   ├── ModelPick.*
│   ├── Parameter.*             # singleton store every UI control commits to
│   └── ViewMode.h              # Facet / Wireframe / Ray / Ray-GS
├── main/main.cpp               # QApplication + QMainWindow hosting a vsgQt render surface
├── main/ToolTracker.h
├── main/StlImporter.*          # ASCII/binary STL reader -> triangle soup
├── main/ThreeMfImporter.*      # 3MF (ZIP + XML) reader -> triangle soup
├── main/profile_hotspots.cpp   # headless timing of cast / sweep / boolean / splat
├── scripts/
│   ├── bootstrap.sh            # macOS: brew + build VSG/vsgQt into ./.deps
│   ├── bootstrap.ps1           # Windows: Vulkan SDK + Qt6 + build VSG/vsgQt into ./.deps
│   ├── run.sh                  # macOS: run with MoltenVK ICD / dylib paths
│   └── run.ps1                 # Windows: run the exe (PATH + Qt plugins)
└── README.md
```

## What it does

`main()`:
1. Starts a `QApplication` and creates a `QMainWindow`.
2. Builds a scene with a single cube via `vsg::Builder::createBox()`.
3. Creates a `vsgQt::Window` (a `QWindow` backed by a VSG/Vulkan surface) and
   embeds it in the main window with `QWidget::createWindowContainer`.
4. Sets up a perspective camera + `vsg::Trackball` and drives rendering through a
   `vsgQt::Viewer`.

Controls: **left-drag** rotate, **right-drag / scroll** zoom, **middle-drag** pan.

### Left panel: model import

The left control panel has **Import STL…** and **Import 3MF…** buttons at the
top. The two differ only in which format their file dialog offers first: the
reader is chosen from the extension of the file actually picked, so either
dialog will take either format, as will `--import`. On click:
1. Opens a file dialog.
2. Reads the file with `app::StlImporter` (ASCII or binary, auto-detected) or
   `app::ThreeMfImporter`, both of which produce an `app::TriangleMesh`.
3. Converts the triangle soup into an `app::BRep` — a boundary representation
   that welds coincident vertices and stores the face→vertex incidence in **CSR**
   form (`faceOffsets` row pointers + `faceVertices` column indices).
4. Builds the face **BVH** over that topology, inside `BRep::fromTriangles`, so a
   `BRep` always arrives ready to cast against.
5. Runs `BRep::validate()`: counts how many faces share each edge. If any edge is
   used by only one face (boundary) or by more than two (non-manifold), the mesh
   is **not watertight** and a warning dialog is shown with the details.
6. Hands the `BRep` to `app::RenderManager`, which builds a phong-shaded VSG
   subgraph and hot-swaps it into the running viewer.

The startup cube takes the same route: it is created as a `BRep` rather than by
`vsg::Builder`, so everything on screen is always backed by a `BRep`.

### Reading 3MF

A 3MF file is an OPC package: a ZIP archive whose model part is XML. The
importer walks the ZIP central directory itself and uses zlib only to inflate
the parts it wants, then reads the XML with `QXmlStreamReader`.

Only geometry is taken. Materials, colours, textures and metadata are ignored,
since everything downstream works from topology alone.

What it does handle:
- **The build section**, flattened. Every `<item>` is instantiated with its
  transform, and `<components>` are followed recursively so an assembly arrives
  as one mesh.
- **The production extension.** Slicers such as Bambu Studio, Orca and
  PrusaSlicer put each object in its own model part and reference it from the
  root with `p:path`. Object ids are numbered per part, so a reference is only
  meaningful together with the part it names; parts are parsed on demand as
  those references are followed. Without this a slicer-written file looks like
  it references objects nothing defines.
- **Units.** Everything is scaled to millimetres, the format's default, so a
  3MF and an STL of the same part arrive the same size.
- Attributes matched on local name, so a namespace prefix cannot hide them.

What it refuses, with a message rather than a guess: ZIP64 and encrypted
archives, compression methods other than stored and deflate, transforms that
are not twelve numbers, triangles indexing vertices that do not exist, and
components nested past a depth cap (which is also what catches a reference
cycle).

### Parameter store

`app::Parameter` is a singleton created as the application starts, reached with
`app::Parameter::instance()`. Every panel control commits its value there first,
and the code that acts on a setting reads it back from the store rather than
from the widget, so the widgets stay pure input and there is one place to look
for the current state. Command-line values are committed the same way.

It currently holds the view mode, the per-axis ray resolution, the ray
divisions used to seed that resolution, the continuous-update flag and the last
imported file path.

`RenderManager` deliberately does not read the store — it is handed the values
it needs, so it stays usable outside this application.

### Left panel: view mode

Below **Reset View** a **View Mode** dropdown selects how the current `BRep` is
drawn. Changing it rebuilds the geometry from the stored topology, so it applies
to the startup cube and to any imported model.

| Mode | Drawn as |
|------|----------|
| Facet | shaded triangles (`faceVertices` as a triangle list) |
| Wireframe | `BRep::extractEdgeIndices()` as a line list — each undirected edge once, back-face culling off |
| Ray | the `RayModel` spans as a line list, coloured by axis (x red, y green, z blue) |
| Ray-GS | the endpoints of those spans as Gaussian splats, shaded as grey metal |

### RayModel

Selecting **Ray** or **Ray-GS** casts an axis-aligned grid of rays through the
current `BRep` and draws the stretches of solid material they pass through.

- `Point3d` (`lib/geom/Ray.h`) is `std::array<double, 3>`; a `Ray` is a
  `startPoint`/`endPoint` pair and a `RayChain` is the `std::vector<Ray>`
  produced by one cast.
- `BoundingBox` (`lib/geom/BoundingBox.h`) is the axis-aligned extent the grid is
  laid out over, built with `BoundingBox::fromBRep`.
- `RayModel::fromBRep(brep, resolution)` holds a `std::vector<RayChain>` for
  each of the x, y and z directions plus the bounding box and the per-axis
  tessellation resolution.

For each direction the grid steps over the two perpendicular axes in cyclic
order, starting at the bounding box minimum: casting along **x** sweeps **y**
from `minY` to `maxY`, then steps **z** and repeats the y loop. Every cast
collects its intersections with the surface, sorts them and pairs them up — 1st
and 2nd form the first `Ray`, 3rd and 4th the second, and so on — so a cast
through two separate lumps of material yields a chain of two rays. Casts that
miss the model produce no chain.

Because the rays are axis aligned, intersection reduces to a point-in-triangle
test in the plane perpendicular to the cast, which also makes hits on shared
edges and vertices easy to merge.

A cast only tests the triangles the **BVH** hands it. The same axis alignment
that simplifies the intersection also simplifies the lookup: a cast is an
infinite line along one axis, so the traversal ignores that axis and descends
only into nodes whose footprint in the other two contains the sample point.
Node bounds are floats taken straight from the vertex positions, which makes
them exact rather than conservative — the hierarchy can only reject a triangle
the per-triangle test would have rejected anyway, so the rays come out bit for
bit identical to a full scan. On a 640k-triangle mesh this turns an 8.1 s cast
into 6.8 ms, against a one-off 40 ms to build the hierarchy at import. Meshes of
a dozen triangles get no benefit, and none is needed. `collectHits` still falls
back to scanning every face if it is ever handed a `BRep` with no hierarchy.

Casts do not depend on one another, so for each direction the two nested sweeps
are flattened into a single range and run through `tbb::parallel_for` — flat
rather than nested so the work still divides evenly when one axis has only a few
steps. Each chain is tagged with the cast that produced it and the results are
sorted back into grid order at the end, which keeps the output identical to a
serial run rather than varying with how the range happened to be split.

The `RenderManager` keeps the models it has been given, keyed by the resolution
they were cast at, so returning to a resolution that has already been cast
redraws from the stored model instead of casting again — switching between
Facet, Ray and Ray-GS is free after the first visit to each resolution. Only the
eight most recently used resolutions are kept, since a cast model can run to
tens of megabytes. The rays belong to the `BRep` they were cast through, so
`showBRep` drops all of them: importing a new STL starts over.

### Gaussian splatting (Ray-GS)

**Ray-GS** draws the same rays as a point cloud: every `startPoint` and
`endPoint` becomes one splat, so the result is the sampled surface of the model
rather than its interior spans.

`createGaussianSplatNode` (`lib/graphics/GaussianSplat.h`) builds the subgraph from a
list of positions, colours and radii. Each splat is a quad that the vertex
shader positions by transforming the centre into eye space and offsetting the
corners there, which turns it to face the camera; the fragment shader evaluates
an isotropic Gaussian over the quad and discards outside it. The shaders are
plain GLSL compiled by VSG when the pipeline is first compiled.

The set is drawn twice, sharing one set of buffers:

1. a depth-only pass that fixes the near surface, laying it down half a splat
   further away than it really is, and
2. a blended pass that composites the Gaussians source-over without writing
   depth.

The first pass draws only each splat's core, out to half the quad, which is
still far enough to reach the corners of its grid cell and so leaves the near
surface sealed. Letting the soft rim into the depth buffer instead stamps a hard
disc over the neighbouring splats, which is what makes the surface break up into
visible circles when zoomed in. The colour pass covers the whole quad and uses a
windowed Gaussian — the falloff's value at the quad edge is subtracted off and
the rest rescaled — so a splat fades to exactly zero rather than ending on a
rim.

The half-splat offset in the first pass is what lets every splat sampling the
near surface contribute to the second, while still rejecting the far side.
Drawing in a single pass instead leaves each splat's soft rim writing depth and
blocking its neighbours, which shows up as dark scalloped edges rather than a
continuous sheet. Compositing is order dependent and is not corrected by a
per-frame sort, but it only matters between splats sampling the same surface.

Unlike Ray, every splat carries the same albedo — a cool grey
(`RenderManager::_splatColor`) — so that where splats overlap the pixel mixes
shades of one grey rather than blending the three axis colours into a muddle. A
single splat is well short of opaque (`RenderManager::_splatOpacity`), so how
many of them cover a pixel comes through as how solid it reads.

The colour pass then shades that grey as polished steel, driven by a per-splat
**surface normal** recorded during the cast. `collectHits` knows which triangle
each crossing belongs to, so it takes that face's normal from the edge cross
product and carries it on the `Hit`. `Ray` stores one for each end
(`startNormal`, `endNormal`) as floats, since a normal only ever feeds shading.

Triangle winding decides the sign of a cross product and an STL is under no
obligation to get it consistently right, so the normals are not trusted as they
come. A ray travels up its axis, entering the solid at `startPoint` and leaving
at `endPoint`, so at the start the outward normal must oppose the axis and at the
end it must follow it; `orientOutward` flips whichever disagrees. That makes the
result independent of the winding.

Each splat is then flat-shaded by its own normal, with no dome perturbation at
all. That is deliberate, and it is what stops splats reading as little spheres:
a radial gradient in the normal becomes a radial gradient in the highlight,
which is exactly what a lit ball looks like. Two earlier approaches are worth
recording as dead ends:

- Shading from per-splat domes alone cannot light a surface. At any useful
  density every pixel is covered by some splat's highlight, so the highlights
  average into a uniform sheen however tight or bright they are made.
- Taking the normal from the ray axis instead gives only six directions. It does
  produce form shading, but the shading steps between the six show up as
  concentric bands, and a highlight cannot be made tight because six normals
  cannot line up closely with the halfway vector.

Two things about the levels are worth knowing before tuning them. The ambient
term is far lower than looks reasonable in the source, because the render target
applies an sRGB transfer curve on write which lifts midtones hard — an ambient
that looks comfortable here renders as pale fog. And no rim or grazing term is
used, because a splat's edge is not a silhouette, so lighting it outlines every
splat and frosts the whole surface.

### Zooming in

A splat's radius is fixed in the space the model is fitted into, so zooming in
magnifies the splat grid until each splat's own outline resolves and the surface
breaks up. The vertex shader compensates: it works out how much of the viewport
a splat spans, from the projection's vertical scale over the eye depth, and once
that passes a threshold it grows the splat by up to 90% and broadens the
Gaussian bell from 2.77 to 1.35. Both widen the averaging kernel, so a splat's
own profile stops being something the eye can pick out. Growth is ramped with
`smoothstep` so the transition is not a visible step, and the depth pre-pass
offset uses the grown radius so it keeps pace with the quad.

The ramp bounds are set against what a splat actually spans, not guessed: a
default view of a model sits near 0.09, so the ramp runs from 0.10 to 0.32. This
is easy to get wrong by an order of magnitude, in which case the ramp sits
saturated at every zoom level and appears to do nothing. Because the measure is
the splat's own apparent size, a finer ray resolution rides further up the zoom
range before it flattens, which is the right behaviour — smaller splats resolve
later.

### Sizing the splats

Rays running along one axis are cast on a grid over the *other two*, so it is
those two resolutions that decide how far apart the endpoints land — which is
why `RenderManager::splatRadius` takes an axis and the three ray families end up
with different radii. Half the grid cell's diagonal is the smallest radius that
still reaches the cell's corners, i.e. the point at which the splats stop
leaving gaps; the quad is then sized at twice that, because the shader puts half
intensity at half the quad, so neighbouring splats cross at half intensity and
add up to an unbroken sheet.

Because the corners are offset *after* the model-view transform, `splatRadius`
also divides out the scale that fits the model into a unit box.

The panel's **Ray Resolution** X/Y/Z fields hold the grid spacing along each
axis, and **Apply** draws the model at whatever is entered, casting it if that
resolution has not been cast yet (switching the view to Ray unless Ray or Ray-GS
is already showing, in which case it stays where it is). The fields are seeded
whenever a model is loaded with `--ray-divisions` steps across its bounding box
(default 1600), so the starting density suits the model's own size; each axis can
then be tuned independently. A resolution fine enough to need more than four
million casts in one direction is refused with a message rather than left to
run.

### Fine geometry, thinned for display

The default resolution is deliberately finer than anything worth drawing: 1600
steps across the bounding box puts a small part into the millions of rays, which
is what makes the sampled solid an accurate stand-in for the real one. Handing
that to the GPU is another matter, since a splat quad per ray endpoint would run
to gigabytes of vertices.

So the two are separated. The cast stays fine, and the renderer draws only every
`stride`'th grid line in each direction, `stride` being the smallest one that
brings the drawn rays under `maxRenderedRays`. Splat radius is multiplied by the
same stride, because endpoints that are now that much further apart need
correspondingly larger splats to still meet.

The stride is searched one integer at a time rather than doubled. That costs an
extra counting pass or two but keeps the drawn density just under the budget at
every cast resolution, so asking for a finer cast never comes back as a coarser
picture — which is what doubling would do each time it crossed a power of two.

Dropping whole layers is what makes this a selection rather than a resampling.
Grid line `i` at stride `s` of an `N`-division cast sits at `lo + s*(extent/N)*i`,
which is where an `N/s`-division cast puts its line `i`, and the intersections
come from the same BVH either way. Every drawn point is a true surface point from
the fine cast; none of them is averaged or interpolated.

How exactly that matches a native coarser cast was measured. At power-of-two
strides it is bit-identical, because scaling a division by a power of two is
exact in binary floating point: a 1600-division cast at stride 4 and a native
400-division cast agree on every one of 1.7 million coordinates. At other strides
`extent/1600*5` and `extent/320` differ in the last bits, so positions match to
about 1e-14 of the model's size and roughly 0.2% of casts — ones grazing the
surface — resolve a hit differently. Neither is visible, and the fine cast's
answer is the better-resolved one.

The practical effect on a 25k-triangle torus:

| divisions | rays cast | cast time | model in RAM | stride | rays drawn | GPU vertices |
|---|---|---|---|---|---|---|
| 160 | 91k | 14 ms | 31 MB | 1 | 91k | 36 MB |
| 400 | 570k | 41 ms | 82 MB | 1 | 570k | 226 MB |
| 800 | 2.3M | 111 ms | 250 MB | 2 | 570k | 226 MB |
| 1600 | 9.1M | 438 ms | 924 MB | 4 | 570k | 226 MB |

The cast keeps getting finer while the GPU load stays flat. Because the BVH makes
casting nearly independent of mesh complexity, a 638k-triangle model costs about
the same as this one.

Holding several such models would not fit, so the cache is bounded by the rays it
holds (`maxCachedRays`) as well as by how many models. At the finest resolutions
that leaves room for one, and the previous cast is dropped as a new one arrives;
coarse models still cache up to `maxCachedRayModels`.

**Apply** casts the resolution exactly as it was typed, and nothing coarsens it
behind your back. Because the renderer copes with any density, being too fine to
*draw* is never a reason to change a resolution — that is what the stride is for.

The one ceiling is memory, since the spans have to be held: `maxCastsPerDirection`
allows eight million casts in a direction, about 2.8 GB on a part the rays cross a
couple of times, or roughly 2800 divisions. Past it the cast is refused rather than
silently substituted, and the message names a resolution that would fit —
`RayModel::finestCastableResolution` scales all three spacings by one factor,
keeping the ratio between the axes — leaving the choice of what to type with you.

The practical range on the torus, all at the resolution asked for:

| divisions | rays cast | model in RAM | stride | rays drawn |
|---|---|---|---|---|
| 400 | 570k | 82 MB | 1 | 570k |
| 1600 | 9.1M | 923 MB | 4 | 570k |
| 2000 | 14.3M | 1430 MB | 5 | 570k |
| 2800 | 28.0M | 2777 MB | 7 | 570k |

### RenderManager

`app::RenderManager` is the handshake between application geometry and the VSG
scene. It holds the scene root Group referenced by the viewer's command graph and:

- `createNode(brep)` — converts CSR topology into a `VertexIndexDraw` with
  positions, area-weighted per-vertex normals, texcoords and an instance colour,
  wrapped in a `StateGroup` built from `vsg::createPhongShaderSet`. Optionally
  fits the model into a unit box at the origin (`setFitToUnitBox`).
- `showBRep(brep)` / `addBRep(brep)` / `clear()` — attach geometry to the scene.
  Because the viewer is already running, each new subgraph is compiled via
  `viewer->compileManager` and applied with `vsg::updateViewer` before being added.
- `setViewMode(mode)` — rebuilds the last shown `BRep` as facets or wireframe.
  The `BRep` is retained for exactly this reason: geometry is regenerated from
  the source topology rather than mutated in the scene graph.
- `setRayModel(model)` / `useCachedRayModel(resolution)` — hand over a freshly
  cast model, or redraw one already held for that resolution. The caller only
  casts when `useCachedRayModel` returns false.

### Command-line options

| Flag | Description |
|------|-------------|
| `--window W H` / `-w W H` | window size (default 1024x768) |
| `--samples N` | MSAA sample count |
| `--import FILE` | import an `.stl` or `.3mf` immediately at startup |
| `--view-mode MODE` | start in `facet` (default), `wireframe`, `ray` or `ray-gs` |
| `--ray-divisions N` | ray grid steps across the bounding box (default 1600) |
| `--debug` / `-d` | enable Vulkan validation layer |
| `--api` / `-a` | enable Vulkan API dump layer |

## Build & run (macOS, Apple Silicon)

Dependencies (Qt6, Vulkan loader/headers, MoltenVK, oneTBB) are installed via
Homebrew; VSG and vsgQt are built from pinned tags into `./.deps`. zlib, which
the 3MF importer uses, is already part of the macOS SDK.

```bash
./scripts/bootstrap.sh

cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="$PWD/.deps;$(brew --prefix qt);$(brew --prefix vulkan-headers);$(brew --prefix vulkan-loader)"
cmake --build build -j"$(sysctl -n hw.ncpu)"

./scripts/run.sh
```

To put the executable and its dylibs in a copyable `dist/bin` folder:

```bash
./scripts/package.sh
./dist/run.sh
```

Copy the whole `dist/` folder to another Apple Silicon Mac. That machine still needs Homebrew Qt (`brew install qt`); Vulkan, TBB, and MoltenVK are included in `dist/bin`. Release builds use `-march=native`, so stay on Apple Silicon.

## Build & run (Windows)

Prerequisites:

- Visual Studio 2019/2022 (or Build Tools) with the **Desktop development with C++** workload
- [CMake](https://cmake.org/) and [Git](https://git-scm.com/)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) (sets `VULKAN_SDK`)
- Qt 6 MSVC 64-bit — used from `C:\Qt\...` if present, otherwise `bootstrap.ps1` fetches it with [aqtinstall](https://github.com/miurahr/aqtinstall) (requires Python)
- [oneTBB](https://github.com/uxlfoundation/oneTBB) — `bootstrap.ps1` does not fetch it yet, so install it and add its prefix to `CMAKE_PREFIX_PATH`
- zlib — needed by the 3MF importer; if CMake cannot find one, install it (vcpkg's `zlib` works) and add its prefix to `CMAKE_PREFIX_PATH`

```powershell
.\scripts\bootstrap.ps1

cmake -S . -B build -A x64 `
  -DCMAKE_PREFIX_PATH="$PWD\.deps;<Qt msvc*_64 prefix>;$env:VULKAN_SDK"
cmake --build build --config Release --parallel

.\scripts\run.ps1
```

`bootstrap.ps1` prints the exact `CMAKE_PREFIX_PATH` to use after it finishes.
`bootstrap.cmd` / `run.cmd` are cmd.exe wrappers around the same scripts.

On Windows the CMake build copies `vsg.dll` / `vsgQt.dll` next to the exe and
runs `windeployqt` so the Qt `qwindows` platform plugin is available.

## Pinned versions

- VulkanSceneGraph `v1.1.16`
- vsgQt `v0.5.0` (built against Qt6)

## macOS note

VSG's stock macOS surface uses the deprecated `VK_MVK_macos_surface` path. When
the VSG window is embedded in a host-owned `NSView` (as Qt does), that path does
not expose a `CAMetalLayer`, so MoltenVK reports no present-capable queue and
window creation fails with *"no suitable Vulkan PhysicalDevice available"*.

`bootstrap.sh` applies [`patches/vsg-macos-metal-surface.patch`](patches/vsg-macos-metal-surface.patch),
which switches VSG to the modern `VK_EXT_metal_surface` (`vkCreateMetalSurfaceEXT`)
and hosts a `CAMetalLayer` sublayer inside the embedded Qt view.
