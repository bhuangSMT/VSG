#!/usr/bin/env bash
#
# Copies the built app and its first-party / VSG / Vulkan runtime dylibs into
# dist/bin so the folder can be copied to another Apple Silicon Mac.
#
# Qt stays a Homebrew install on the other machine:
#   brew install qt
# Then:
#   ./dist/run.sh
#
# Release is built with -march=native, so the other Mac should be Apple Silicon.

set -euo pipefail

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "error: package.sh is for macOS." >&2
        exit 1
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
DEPS="$ROOT/.deps"
DIST="$ROOT/dist"
BIN="$DIST/bin"

if [ ! -x "$BUILD/vsg_qt_cube" ]; then
    echo "error: $BUILD/vsg_qt_cube is missing. Build the app first." >&2
    exit 1
fi

if [ -d /opt/homebrew/opt/molten-vk ]; then
    MOLTENVK="/opt/homebrew/opt/molten-vk"
    VK_LOADER="/opt/homebrew/opt/vulkan-loader"
    TBB="/opt/homebrew/opt/tbb"
    GLSLANG="/opt/homebrew/opt/glslang"
    SPIRV_TOOLS="/opt/homebrew/opt/spirv-tools"
else
    MOLTENVK="$(brew --prefix molten-vk)"
    VK_LOADER="$(brew --prefix vulkan-loader)"
    TBB="$(brew --prefix tbb)"
    GLSLANG="$(brew --prefix glslang)"
    SPIRV_TOOLS="$(brew --prefix spirv-tools)"
fi

rm -rf "$DIST"
mkdir -p "$BIN" "$DIST/vulkan/icd.d"

copy_one() {
    local src="$1"
    local dest="$BIN/$(basename "$src")"
    if [ ! -e "$dest" ]; then
        cp -a "$src" "$BIN/"
    fi
    if [ -L "$src" ]; then
        local target
        target="$(readlink "$src")"
        if [[ "$target" != /* ]]; then
            target="$(dirname "$src")/$target"
        fi
        if [ -e "$target" ]; then
            copy_one "$target"
        fi
    fi
}

copy_dylibs() {
    local src="$1"
    shift
    local name
    for name in "$@"; do
        local matches=("$src"/$name)
        if [ ! -e "${matches[0]}" ]; then
            echo "error: no match for $src/$name" >&2
            exit 1
        fi
        local file
        for file in "${matches[@]}"; do
            copy_one "$file"
        done
    done
}

cp -a "$BUILD/vsg_qt_cube" "$BIN/"
copy_dylibs "$BUILD" "libucam_geom*.dylib" "libucam_boolean*.dylib" "libucam_graphics*.dylib"
copy_dylibs "$DEPS/lib" "libvsg*.dylib" "libvsgQt*.dylib"
copy_dylibs "$TBB/lib" "libtbb.12.dylib" "libtbb.dylib"
copy_dylibs "$VK_LOADER/lib" "libvulkan.1.dylib" "libvulkan.dylib"
copy_dylibs "$MOLTENVK/lib" "libMoltenVK.dylib"
copy_dylibs "$GLSLANG/lib" \
    "libglslang.16.dylib" \
    "libSPIRV.16.dylib" \
    "libglslang-default-resource-limits.16.dylib"
copy_dylibs "$SPIRV_TOOLS/lib" "libSPIRV-Tools.dylib" "libSPIRV-Tools-opt.dylib"

cat > "$DIST/vulkan/icd.d/MoltenVK_icd.json" <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "../../bin/libMoltenVK.dylib",
        "api_version": "1.4.0",
        "is_portability_driver": true
    }
}
EOF

rewrite_deps() {
    local file="$1"
    local dep
    while IFS= read -r dep; do
        case "$dep" in
            /System/*|/usr/lib/*|@rpath/*|@loader_path/*) ;;
            *Qt*.framework*) ;;
            *.dylib)
                install_name_tool -change "$dep" "@rpath/$(basename "$dep")" "$file"
                ;;
        esac
    done < <(otool -L "$file" | awk 'NR > 1 { print $1 }')
}

ensure_rpath() {
    local file="$1"
    if ! otool -l "$file" | grep -q '@loader_path'; then
        install_name_tool -add_rpath '@loader_path' "$file"
    fi
}

shopt -s nullglob
for file in "$BIN/vsg_qt_cube" "$BIN"/*.dylib; do
    if [ -L "$file" ]; then
        continue
    fi
    chmod u+w "$file"
    if [[ "$file" == *.dylib ]]; then
        install_name_tool -id "@rpath/$(basename "$file")" "$file"
    fi
    rewrite_deps "$file"
    ensure_rpath "$file"
    codesign --force -s - "$file" >/dev/null
done
shopt -u nullglob

cat > "$DIST/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="$ROOT/vulkan/icd.d/MoltenVK_icd.json"
export DYLD_LIBRARY_PATH="$ROOT/bin${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$ROOT/bin/vsg_qt_cube" "$@"
EOF
chmod +x "$DIST/run.sh"

echo "Packaged $DIST"
echo "  Copy that folder to another Apple Silicon Mac, install Qt there:"
echo "    brew install qt"
echo "  then run:"
echo "    ./dist/run.sh"
