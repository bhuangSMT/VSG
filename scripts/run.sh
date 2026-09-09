#!/usr/bin/env bash
#
# Runs the built cube app with the environment MoltenVK needs on macOS:
#  - VK_ICD_FILENAMES points the Vulkan loader at the MoltenVK driver
#  - DYLD_LIBRARY_PATH lets the app find the locally-built VSG/vsgQt dylibs

set -euo pipefail

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "error: use scripts/run.ps1 (or scripts/run.cmd) on Windows." >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS="$ROOT/.deps"

# Prefer the Homebrew keg paths directly so launch does not wait on `brew`.
if [ -d /opt/homebrew/opt/molten-vk ]; then
    MOLTENVK="/opt/homebrew/opt/molten-vk"
    VK_LOADER="/opt/homebrew/opt/vulkan-loader"
else
    MOLTENVK="$(brew --prefix molten-vk)"
    VK_LOADER="$(brew --prefix vulkan-loader)"
fi

export VK_ICD_FILENAMES="$MOLTENVK/etc/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$DEPS/lib:$VK_LOADER/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

exec "$ROOT/build/vsg_qt_cube" "$@"
