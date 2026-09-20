# Build and zip the C++ SDK on Windows: include/ + lib/ + bin/.
# Writes dist/ucam-sdk-windows-x64.zip
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "CMakeLists.txt"))) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$Build = if ($env:UCAM_BUILD_DIR) { $env:UCAM_BUILD_DIR } else { Join-Path $Root "build" }
$Dist = Join-Path $Root "dist"
$Stage = Join-Path $Dist "ucam-sdk-windows-x64"
$Zip = Join-Path $Dist "ucam-sdk-windows-x64.zip"

cmake -S $Root -B $Build -DUCAM_BUILD_SDK=ON -DUCAM_NATIVE_ARCH=OFF
cmake --build $Build --config Release --target ucam ucam_sdk_example ucam_sdk_view_preview -j8

if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
cmake --install $Build --prefix $Stage --component ucam_sdk --config Release

$Lib = Join-Path $Stage "lib"
$Bin = Join-Path $Stage "bin"
New-Item -ItemType Directory -Force -Path $Lib, $Bin | Out-Null

function Copy-IfExists($src, $destDir) {
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        Copy-Item $src $destDir -Force
    }
}

Get-ChildItem $Build -Filter "ucam*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName $Bin -Force
}
Get-ChildItem $Build -Filter "ucam*.lib" -ErrorAction SilentlyContinue | ForEach-Object {
    Copy-Item $_.FullName $Lib -Force
}

$Windeploy = $null
Get-ChildItem env: | Out-Null
$qtHint = @()
if ($env:CMAKE_PREFIX_PATH) { $qtHint += $env:CMAKE_PREFIX_PATH.Split(";") }
foreach ($hint in $qtHint) {
    $cand = Join-Path $hint "bin\windeployqt.exe"
    if (Test-Path $cand) { $Windeploy = $cand; break }
}
if (-not $Windeploy) {
    $Windeploy = Get-Command windeployqt -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
}

$UcamDll = Join-Path $Bin "ucam.dll"
if (-not (Test-Path $UcamDll)) {
    $UcamDll = Join-Path $Build "ucam.dll"
}
if ($Windeploy -and (Test-Path $UcamDll)) {
    & $Windeploy --dir $Bin --libdir $Bin --plugindir (Join-Path $Bin "platforms") --no-translations --no-compiler-runtime $UcamDll
}

foreach ($name in @("vsg.dll", "vsgd.dll", "vsgQt.dll", "tbb.dll", "tbb12.dll", "vulkan-1.dll", "zlib.dll", "zlib1.dll")) {
    $found = Get-ChildItem $Build, (Join-Path $Root ".deps\bin"), (Join-Path $Root ".deps\lib") -Filter $name -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName $Bin -Force
        $libName = [IO.Path]::ChangeExtension($found.Name, ".lib")
        $libFound = Get-ChildItem $Build, (Join-Path $Root ".deps\lib") -Filter $libName -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($libFound) { Copy-Item $libFound.FullName $Lib -Force }
    }
}

if (Test-Path $Zip) { Remove-Item $Zip -Force }
Compress-Archive -Path $Stage -DestinationPath $Zip
Write-Host "SDK zip: $Zip"
