# Installs native dependencies and builds VulkanSceneGraph + vsgQt into ./.deps
# so that the top-level CMake project can find them via CMAKE_PREFIX_PATH.
#
# Safe to re-run; already-built dependencies are reused.
#
# Prerequisites that this script will try to locate (and install when possible):
#   - CMake, Git, a Visual Studio C++ toolchain
#   - Vulkan SDK (VULKAN_SDK or C:\VulkanSDK\*)
#   - Qt6 (official install, or fetched via aqtinstall into .deps/qt)
#   - oneTBB and zlib (cloned and installed into .deps)

[CmdletBinding()]
param(
    [string]$QtPrefix = "",
    [string]$VulkanSdk = $env:VULKAN_SDK
)

$ErrorActionPreference = "Stop"

$VSG_TAG = "v1.1.16"
$VSGQT_TAG = "v0.5.0"
$TBB_TAG = "v2022.1.0"
$ZLIB_TAG = "v1.3.1"
$QT_VERSION = "6.8.3"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Deps = Join-Path $Root ".deps"
$Src = Join-Path $Deps "src"
New-Item -ItemType Directory -Force -Path $Src | Out-Null

function Find-VsWhere {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

function Get-VsCMakeGenerator {
    $vswhere = Find-VsWhere
    if (-not $vswhere) { return $null }
    $year = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property catalog_productLineVersion 2>$null
    switch ("$year".Trim()) {
        "2019" { return "Visual Studio 16 2019" }
        "2022" { return "Visual Studio 17 2022" }
        "2025" { return "Visual Studio 18 2025" }
        default { return $null }
    }
}

function Get-QtArch {
    $vswhere = Find-VsWhere
    if (-not $vswhere) { return "win64_msvc2022_64" }
    $year = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property catalog_productLineVersion 2>$null
    if ("$year".Trim() -eq "2019") { return "win64_msvc2019_64" }
    return "win64_msvc2022_64"
}

function Find-QtPrefix {
    param([string]$Hint)

    $hints = @()
    if ($Hint) { $hints += $Hint }
    if ($env:QTDIR) { $hints += $env:QTDIR }
    if ($env:Qt6_DIR) {
        try {
            $hints += (Resolve-Path (Join-Path $env:Qt6_DIR "../../..")).Path
        } catch { }
    }

    $searchRoots = @(
        (Join-Path $Deps "qt"),
        "C:\Qt"
    )

    foreach ($h in $hints) {
        if ($h -and (Test-Path (Join-Path $h "lib\cmake\Qt6\Qt6Config.cmake"))) { return $h }
        if ($h -and (Test-Path (Join-Path $h "lib\cmake\Qt5\Qt5Config.cmake"))) { return $h }
    }

    foreach ($root in $searchRoots) {
        if (-not (Test-Path $root)) { continue }
        $found = Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object {
                Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue |
                    Where-Object { $_.Name -match '^msvc' }
            }
        foreach ($msvc in $found) {
            if (Test-Path (Join-Path $msvc.FullName "lib\cmake\Qt6\Qt6Config.cmake")) {
                return $msvc.FullName
            }
        }
    }
    return $null
}

function Find-VulkanSdk {
    param([string]$Hint)
    if ($Hint -and (Test-Path (Join-Path $Hint "Include\vulkan\vulkan.h"))) { return $Hint }
    if ($env:VULKAN_SDK -and (Test-Path (Join-Path $env:VULKAN_SDK "Include\vulkan\vulkan.h"))) {
        return $env:VULKAN_SDK
    }
    $sdkRoot = "C:\VulkanSDK"
    if (Test-Path $sdkRoot) {
        $latest = Get-ChildItem $sdkRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            Select-Object -First 1
        if ($latest -and (Test-Path (Join-Path $latest.FullName "Include\vulkan\vulkan.h"))) {
            return $latest.FullName
        }
    }
    return $null
}

function Get-Python {
    foreach ($cmd in @("py", "python", "python3")) {
        $exe = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($exe) { return $exe.Source }
    }
    return $null
}

function Invoke-CMakeConfigure {
    param(
        [string]$Source,
        [string]$Build,
        [string[]]$Arguments
    )
    $cfg = @("-S", $Source, "-B", $Build) + $Arguments
    $cache = Join-Path $Build "CMakeCache.txt"
    if (-not (Test-Path $cache)) {
        $gen = Get-VsCMakeGenerator
        if ($gen) {
            $cfg = @("-G", $gen, "-A", "x64") + $cfg
        }
    }
    & cmake @cfg
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed for $Source" }
}

function Invoke-CMakeInstall {
    param([string]$Build)
    & cmake --build $Build --config Release --target install --parallel
    if ($LASTEXITCODE -ne 0) { throw "cmake build/install failed for $Build" }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is required. Install from https://cmake.org/ or: winget install Kitware.CMake"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install from https://git-scm.com/ or: winget install Git.Git"
}

Write-Host "==> Locating Vulkan SDK"
$VulkanSdk = Find-VulkanSdk -Hint $VulkanSdk
if (-not $VulkanSdk) {
    Write-Host "    Vulkan SDK not found. Attempting winget install (KhronosGroup.VulkanSDK)..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id KhronosGroup.VulkanSDK -e --accept-package-agreements --accept-source-agreements
        $VulkanSdk = Find-VulkanSdk -Hint $env:VULKAN_SDK
    }
}
if (-not $VulkanSdk) {
    throw @"
Vulkan SDK not found. Install it from https://vulkan.lunarg.com/sdk/home
then re-run this script (or set VULKAN_SDK to the install directory).
"@
}
$env:VULKAN_SDK = $VulkanSdk
Write-Host "    $VulkanSdk"

Write-Host "==> Locating Qt6"
$QtPrefix = Find-QtPrefix -Hint $QtPrefix
if (-not $QtPrefix) {
    $python = Get-Python
    if (-not $python) {
        throw @"
Qt6 not found and Python is not available to fetch it via aqtinstall.

Install Qt 6 (MSVC 64-bit) from https://www.qt.io/download-qt-installer
or install Python and re-run this script.
"@
    }
    $arch = Get-QtArch
    $qtDest = Join-Path $Deps "qt"
    Write-Host "    Fetching Qt $QT_VERSION ($arch) into $qtDest via aqtinstall"
    & $python -m pip install --user --upgrade aqtinstall
    if ($LASTEXITCODE -ne 0) { throw "pip install aqtinstall failed" }
    & $python -m aqt install-qt windows desktop $QT_VERSION $arch --outputdir $qtDest
    if ($LASTEXITCODE -ne 0) { throw "aqt install-qt failed" }
    $QtPrefix = Find-QtPrefix
}
if (-not $QtPrefix) {
    throw "Qt6 was installed but cmake config was not found. Pass -QtPrefix <msvc*_64 dir>."
}
Write-Host "    $QtPrefix"

$env:CMAKE_PREFIX_PATH = ($Deps, $QtPrefix, $VulkanSdk, $env:CMAKE_PREFIX_PATH | Where-Object { $_ }) -join ";"

$cmakeCommon = @(
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$Deps",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCMAKE_PREFIX_PATH=$($env:CMAKE_PREFIX_PATH)"
)

$tbbSrc = Join-Path $Src "oneTBB"
if (-not (Test-Path (Join-Path $tbbSrc ".git"))) {
    Write-Host "==> Cloning oneTBB $TBB_TAG"
    git clone --depth 1 --branch $TBB_TAG https://github.com/uxlfoundation/oneTBB.git $tbbSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone oneTBB failed" }
}
Write-Host "==> Building oneTBB $TBB_TAG"
Invoke-CMakeConfigure -Source $tbbSrc -Build (Join-Path $tbbSrc "build") -Arguments ($cmakeCommon + @("-DTBB_TEST=OFF"))
Invoke-CMakeInstall -Build (Join-Path $tbbSrc "build")

$zlibSrc = Join-Path $Src "zlib"
if (-not (Test-Path (Join-Path $zlibSrc ".git"))) {
    Write-Host "==> Cloning zlib $ZLIB_TAG"
    git clone --depth 1 --branch $ZLIB_TAG https://github.com/madler/zlib.git $zlibSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone zlib failed" }
}
Write-Host "==> Building zlib $ZLIB_TAG"
Invoke-CMakeConfigure -Source $zlibSrc -Build (Join-Path $zlibSrc "build") -Arguments $cmakeCommon
Invoke-CMakeInstall -Build (Join-Path $zlibSrc "build")

$vsgSrc = Join-Path $Src "VulkanSceneGraph"
if (-not (Test-Path (Join-Path $vsgSrc ".git"))) {
    Write-Host "==> Cloning VulkanSceneGraph $VSG_TAG"
    git clone --depth 1 --branch $VSG_TAG https://github.com/vsg-dev/VulkanSceneGraph.git $vsgSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone VulkanSceneGraph failed" }
}
Write-Host "==> Building VulkanSceneGraph $VSG_TAG"
Invoke-CMakeConfigure -Source $vsgSrc -Build (Join-Path $vsgSrc "build") -Arguments $cmakeCommon
Invoke-CMakeInstall -Build (Join-Path $vsgSrc "build")

$vsgQtSrc = Join-Path $Src "vsgQt"
if (-not (Test-Path (Join-Path $vsgQtSrc ".git"))) {
    Write-Host "==> Cloning vsgQt $VSGQT_TAG"
    git clone --depth 1 --branch $VSGQT_TAG https://github.com/vsg-dev/vsgQt.git $vsgQtSrc
    if ($LASTEXITCODE -ne 0) { throw "git clone vsgQt failed" }
}
Write-Host "==> Building vsgQt $VSGQT_TAG"
$vsgQtArgs = $cmakeCommon + @(
    "-DQT_PACKAGE_NAME=Qt6",
    "-DVSGQT_BUILD_EXAMPLES=OFF"
)
Invoke-CMakeConfigure -Source $vsgQtSrc -Build (Join-Path $vsgQtSrc "build") -Arguments $vsgQtArgs
Invoke-CMakeInstall -Build (Join-Path $vsgQtSrc "build")

Write-Host ""
Write-Host "==> Dependencies installed into: $Deps"
Write-Host "    Now configure the app with:"
Write-Host "        cmake -S . -B build -A x64 -DCMAKE_PREFIX_PATH=`"$Deps;$QtPrefix;$VulkanSdk`""
Write-Host "        cmake --build build --config Release --parallel"
Write-Host "    Then run:"
Write-Host "        .\scripts\run.ps1"
