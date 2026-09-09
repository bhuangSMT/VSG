#!/usr/bin/env bash
#
# Installs native dependencies and builds VulkanSceneGraph + vsgQt into ./.deps
# so that the top-level CMake project can find them via CMAKE_PREFIX_PATH.
#
# macOS / Unix. On Windows use scripts/bootstrap.ps1 (or scripts/bootstrap.cmd).
#
# Safe to re-run; already-built dependencies are reused.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/.deps"
SRC="$DEPS/src"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "error: use scripts/bootstrap.ps1 (or scripts/bootstrap.cmd) on Windows." >&2
        exit 1
        ;;
esac

VSG_TAG="v1.1.16"
VSGQT_TAG="v0.5.0"

mkdir -p "$SRC"

if ! command -v brew >/dev/null 2>&1; then
    echo "error: Homebrew is required (https://brew.sh)." >&2
    exit 1
fi

echo "==> Installing Homebrew packages (Qt6, Vulkan loader/headers, MoltenVK, oneTBB)"
brew install qt vulkan-headers vulkan-loader vulkan-tools molten-vk tbb

BREW="$(brew --prefix)"
QT_PREFIX="$(brew --prefix qt)"
VK_HEADERS="$(brew --prefix vulkan-headers)"
VK_LOADER="$(brew --prefix vulkan-loader)"

export CMAKE_PREFIX_PATH="$DEPS:$QT_PREFIX:$VK_HEADERS:$VK_LOADER:$BREW${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"

NPROC="$(sysctl -n hw.ncpu)"
CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$DEPS"
    -DBUILD_SHARED_LIBS=ON
    -DVulkan_INCLUDE_DIR="$VK_HEADERS/include"
    -DVulkan_LIBRARY="$VK_LOADER/lib/libvulkan.dylib"
)

# --- VulkanSceneGraph ------------------------------------------------------
if [ ! -d "$SRC/VulkanSceneGraph/.git" ]; then
    git clone --depth 1 --branch "$VSG_TAG" \
        https://github.com/vsg-dev/VulkanSceneGraph.git "$SRC/VulkanSceneGraph"

    # macOS fix: VSG's stock macOS surface uses the deprecated VK_MVK_macos_surface
    # path, which fails to expose a CAMetalLayer when embedded in a host-owned
    # NSView (e.g. Qt). MoltenVK then reports no present-capable queue and window
    # creation fails. This patch switches to VK_EXT_metal_surface and hosts a
    # CAMetalLayer sublayer inside the embedded view.
    if [ "$(uname)" = "Darwin" ]; then
        echo "==> Patching VulkanSceneGraph for macOS/Qt embedding"
        git -C "$SRC/VulkanSceneGraph" apply "$ROOT/patches/vsg-macos-metal-surface.patch"
    fi
fi
echo "==> Building VulkanSceneGraph $VSG_TAG"
cmake -S "$SRC/VulkanSceneGraph" -B "$SRC/VulkanSceneGraph/build" "${CMAKE_COMMON[@]}"
cmake --build "$SRC/VulkanSceneGraph/build" --target install -j"$NPROC"

# --- vsgQt -----------------------------------------------------------------
if [ ! -d "$SRC/vsgQt/.git" ]; then
    git clone --depth 1 --branch "$VSGQT_TAG" \
        https://github.com/vsg-dev/vsgQt.git "$SRC/vsgQt"
fi
echo "==> Building vsgQt $VSGQT_TAG"
cmake -S "$SRC/vsgQt" -B "$SRC/vsgQt/build" "${CMAKE_COMMON[@]}" \
    -DQT_PACKAGE_NAME=Qt6 \
    -DVSGQT_BUILD_EXAMPLES=OFF
cmake --build "$SRC/vsgQt/build" --target install -j"$NPROC"

echo ""
echo "==> Dependencies installed into: $DEPS"
echo "    Now configure the app with:"
echo "        cmake -S . -B build -DCMAKE_PREFIX_PATH=\"$DEPS;$QT_PREFIX;$VK_HEADERS;$VK_LOADER\""
echo "        cmake --build build -j$NPROC"
