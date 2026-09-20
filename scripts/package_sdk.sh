#!/usr/bin/env bash
# Build and zip the C++ SDK: include/ + lib/ + bin/.
# macOS → dist/ucam-sdk-macos-arm64.zip
# Windows (Git Bash) → dist/ucam-sdk-windows-x64.zip
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${UCAM_BUILD_DIR:-$ROOT/build}"
DIST="${ROOT}/dist"

OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "$OS" == "Darwin" ]]; then
    TAG="macos-${ARCH}"
elif [[ "$OS" == "MINGW"* || "$OS" == "MSYS"* || "$OS" == "CYGWIN"* ]]; then
    TAG="windows-x64"
else
    TAG="$(echo "$OS" | tr '[:upper:]' '[:lower:]')-${ARCH}"
fi

STAGE="${DIST}/ucam-sdk-${TAG}"
ZIP="${DIST}/ucam-sdk-${TAG}.zip"

cmake -S "$ROOT" -B "$BUILD" -DUCAM_BUILD_SDK=ON -DUCAM_NATIVE_ARCH=OFF
cmake --build "$BUILD" --target ucam ucam_sdk_example ucam_sdk_view_preview -j8

rm -rf "$STAGE"
mkdir -p "$STAGE"
cmake --install "$BUILD" --prefix "$STAGE" --component ucam_sdk

mkdir -p "$STAGE/lib" "$STAGE/bin"

copy_if() {
    local src="$1"
    local dest_dir="$2"
    if [[ -f "$src" ]]; then
        mkdir -p "$dest_dir"
        cp -R "$src" "$dest_dir/"
        # Also copy install_name siblings (libfoo.1.dylib).
        local dir; dir="$(dirname "$src")"
        local base; base="$(basename "$src")"
        local stem="${base%%.*}"
        shopt -s nullglob
        for sib in "$dir"/"$stem"*.dylib "$dir"/"$stem"*.dll "$dir"/"$stem"*.so*; do
            cp -R "$sib" "$dest_dir/" 2>/dev/null || true
        done
        shopt -u nullglob
    fi
}

# In-tree runtimes (libucam loads these).
for tgt in ucam ucam_graphics ucam_boolean ucam_geom; do
    if [[ -f "$BUILD/lib${tgt}.dylib" ]]; then
        copy_if "$BUILD/lib${tgt}.dylib" "$STAGE/lib"
        copy_if "$BUILD/lib${tgt}.dylib" "$STAGE/bin"
    elif [[ -f "$BUILD/${tgt}.dll" ]]; then
        copy_if "$BUILD/${tgt}.dll" "$STAGE/bin"
        copy_if "$BUILD/${tgt}.lib" "$STAGE/lib"
    fi
done

# Recursively collect dylib/framework dependents. Extra args are @rpath search dirs.
collect_deps() {
    local binary="$1"
    local dest="$2"
    shift 2
    python3 - "$binary" "$dest" "$@" <<'PY'
import os, shutil, subprocess, sys

binary = sys.argv[1]
dest = sys.argv[2]
search = [os.path.realpath(p) for p in sys.argv[3:] if os.path.isdir(p)]
os.makedirs(dest, exist_ok=True)
seen_files = set()
seen_fw = set()
skip_prefixes = ("/usr/lib/", "/System/", "/Library/Apple/")

def is_system(path: str) -> bool:
    return any(path.startswith(p) for p in skip_prefixes)

def framework_root(path: str):
    marker = ".framework"
    idx = path.find(marker)
    if idx < 0:
        return None
    return path[: idx + len(marker)]

def copy_file(path: str) -> None:
    path = os.path.realpath(path)
    if path in seen_files or not os.path.isfile(path):
        return
    seen_files.add(path)
    fw = framework_root(path)
    if fw and not is_system(fw):
        name = os.path.basename(fw)
        dest_fw = os.path.join(dest, name)
        if fw not in seen_fw:
            seen_fw.add(fw)
            if os.path.realpath(fw) != os.path.realpath(dest_fw):
                if os.path.exists(dest_fw):
                    shutil.rmtree(dest_fw)
                shutil.copytree(fw, dest_fw, symlinks=True)
        return
    if is_system(path):
        return
    dest_path = os.path.join(dest, os.path.basename(path))
    if os.path.realpath(path) != os.path.realpath(dest_path):
        shutil.copy2(path, dest_path)

def resolve(dep: str, origin: str) -> str:
    if dep.startswith("@rpath/"):
        name = dep[len("@rpath/") :]
        for root in search + [os.path.dirname(origin)]:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return os.path.realpath(cand)
        return ""
    if dep.startswith("@loader_path/"):
        cand = os.path.join(os.path.dirname(origin), dep[len("@loader_path/") :])
        return os.path.realpath(cand) if os.path.isfile(cand) else ""
    if os.path.isfile(dep):
        return os.path.realpath(dep)
    cand = os.path.join(os.path.dirname(origin), os.path.basename(dep))
    return os.path.realpath(cand) if os.path.isfile(cand) else ""

def walk(path: str) -> None:
    if not path or not os.path.isfile(path):
        return
    path = os.path.realpath(path)
    copy_file(path)
    try:
        out = subprocess.check_output(["otool", "-L", path], text=True)
    except (OSError, subprocess.CalledProcessError):
        return
    for line in out.splitlines()[1:]:
        dep = line.strip().split(" (compatibility")[0].strip()
        if not dep or is_system(dep):
            continue
        resolved = resolve(dep, path)
        if resolved and resolved not in seen_files:
            walk(resolved)

walk(os.path.realpath(binary))
PY
}

if [[ "$OS" == "Darwin" ]]; then
    UCAM_DYLIB="$BUILD/libucam.dylib"
    SEARCH=("$BUILD" "$ROOT/.deps/lib")
    collect_deps "$UCAM_DYLIB" "$STAGE/bin" "${SEARCH[@]}"
    collect_deps "$UCAM_DYLIB" "$STAGE/lib" "${SEARCH[@]}"

    QT_PLUGIN="/opt/homebrew/Cellar/qtbase/6.11.2/share/qt/plugins/platforms/libqcocoa.dylib"
    if [[ ! -f "$QT_PLUGIN" ]]; then
        QT_PLUGIN="$(find /opt/homebrew/opt/qtbase /opt/homebrew/Cellar/qtbase -name libqcocoa.dylib 2>/dev/null | head -n 1 || true)"
    fi
    if [[ -n "${QT_PLUGIN:-}" && -f "$QT_PLUGIN" ]]; then
        mkdir -p "$STAGE/bin/plugins/platforms" "$STAGE/bin/platforms"
        cp "$QT_PLUGIN" "$STAGE/bin/plugins/platforms/"
        cp "$QT_PLUGIN" "$STAGE/bin/platforms/"
    fi

    mkdir -p "$STAGE/bin/vulkan/icd.d"
    if [[ -f /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json ]]; then
        cp /opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json "$STAGE/bin/vulkan/icd.d/"
    fi
    if [[ -f /opt/homebrew/lib/libMoltenVK.dylib ]]; then
        cp /opt/homebrew/lib/libMoltenVK.dylib "$STAGE/bin/"
        cp /opt/homebrew/lib/libMoltenVK.dylib "$STAGE/lib/"
    fi

    # Preserve SONAME / @rpath filenames (collect_deps copies the real file).
    for dir in "$STAGE/bin" "$STAGE/lib"; do
        (
            cd "$dir"
            [[ -f libvsg.1.1.16.dylib ]] && ln -sf libvsg.1.1.16.dylib libvsg.17.dylib && ln -sf libvsg.17.dylib libvsg.dylib
            [[ -f libvsgQt.0.5.0.dylib ]] && ln -sf libvsgQt.0.5.0.dylib libvsgQt.3.dylib && ln -sf libvsgQt.3.dylib libvsgQt.dylib
            [[ -f libvulkan.1.4.357.dylib ]] && ln -sf libvulkan.1.4.357.dylib libvulkan.1.dylib && ln -sf libvulkan.1.dylib libvulkan.dylib
            [[ -f libtbb.12.19.dylib ]] && ln -sf libtbb.12.19.dylib libtbb.12.dylib && ln -sf libtbb.12.dylib libtbb.dylib
        )
    done
fi

(
    cd "$DIST"
    rm -f "$(basename "$ZIP")"
    ditto -c -k --sequesterRsrc --keepParent "$(basename "$STAGE")" "$(basename "$ZIP")" 2>/dev/null \
        || zip -r "$(basename "$ZIP")" "$(basename "$STAGE")"
)

echo "SDK zip: $ZIP"
ls -la "$STAGE/include/ucam" "$STAGE/lib" "$STAGE/bin" | head -80
