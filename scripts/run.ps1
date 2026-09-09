# Runs the built cube app on Windows.
# windeployqt (from CMake POST_BUILD) copies Qt plugins next to the exe;
# VSG/vsgQt DLLs are copied there too. PATH is still prepended as a fallback.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Deps = Join-Path $Root ".deps"

$exeCandidates = @(
    (Join-Path $Root "build\vsg_qt_cube.exe"),
    (Join-Path $Root "build\Release\vsg_qt_cube.exe"),
    (Join-Path $Root "build\RelWithDebInfo\vsg_qt_cube.exe")
)
$exe = $exeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $exe) {
    throw "vsg_qt_cube.exe not found. Build the app first (see README.md)."
}

$qtBin = $null
$qtPrefix = Get-ChildItem (Join-Path $Deps "qt") -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending |
    ForEach-Object {
        Get-ChildItem $_.FullName -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^msvc' }
    } |
    Select-Object -First 1
if ($qtPrefix) { $qtBin = Join-Path $qtPrefix.FullName "bin" }
elseif ($env:QTDIR) { $qtBin = Join-Path $env:QTDIR "bin" }

$pathParts = @((Join-Path $Deps "bin"), $qtBin, $env:PATH) | Where-Object { $_ }
$env:PATH = $pathParts -join ";"

if ($env:VULKAN_SDK) {
    $env:PATH = "$(Join-Path $env:VULKAN_SDK 'Bin');$($env:PATH)"
}

& $exe @AppArgs
exit $LASTEXITCODE
