#Requires -Version 5.1
<#
.SYNOPSIS
    Generate all PNG icon sizes and a multi-resolution icon.ico from icon.svg.

.DESCRIPTION
    Rasterizes FastTIFF/icon/icon.svg into PNGs at 16, 32, 48, 64, 128 and 256
    pixels, then packs those PNGs into a single multi-resolution icon.ico. All
    output lands next to the SVG in the icon directory.

    icon.svg is the single source of truth: run this whenever the SVG changes to
    refresh the committed rasters. (In CI these are generated on the fly, so you
    only need this for local builds or if you commit the .ico for build.rs.)

    SVG rendering needs one external rasterizer — the script auto-detects, in
    order of preference:
        1. rsvg-convert   (librsvg — sharpest, install: winget install --id GNOME.librsvg
                           or it ships with Inkscape / Git-for-Windows' usr/bin)
        2. inkscape       (winget install --id Inkscape.Inkscape)
        3. magick         (ImageMagick — winget install --id ImageMagick.ImageMagick)
    Packing the .ico is done natively in PowerShell (no external tool needed):
    each size is embedded as a PNG blob, which Windows Vista+ reads at every
    resolution.

.PARAMETER IconDir
    Directory containing icon.svg and where output is written.
    Defaults to the "FastTIFF/icon" folder resolved relative to this script's
    location (…/FastTIFF/icon), falling back to the current directory.

.PARAMETER SvgName
    Source SVG filename inside IconDir. Default: icon.svg

.PARAMETER Sizes
    Pixel sizes to render. Default: 16, 32, 48, 64, 128, 256
    (Sizes >= 256 are stored in the .ico as PNG; that's the standard modern ICO.)

.PARAMETER PngOnly
    Only emit the PNGs; skip building icon.ico.

.PARAMETER IcoOnly
    Only build icon.ico (still renders the PNGs it needs as inputs, but removes
    any it created that weren't already on disk).

.EXAMPLE
    .\generate-icons.ps1
    Render every size + icon.ico into …/FastTIFF/icon.

.EXAMPLE
    .\generate-icons.ps1 -IconDir . -Sizes 16,32,48,256
    Custom source dir and a reduced size set.

.NOTES
    Colours/transparency are preserved (icons render on a transparent canvas).
#>

[CmdletBinding()]
param(
    [string]   $IconDir,
    [string]   $SvgName = 'icon.svg',
    [int[]]    $Sizes   = @(16, 32, 48, 64, 128, 256),
    [switch]   $PngOnly,
    [switch]   $IcoOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Resolve the icon directory.
# ---------------------------------------------------------------------------
if (-not $IconDir) {
    # Prefer a FastTIFF/icon folder near the script, else fall back to CWD.
    $scriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $guess = Join-Path $scriptRoot 'FastTIFF\icon'
    if (Test-Path -LiteralPath (Join-Path $guess $SvgName)) {
        $IconDir = $guess
    } elseif (Test-Path -LiteralPath (Join-Path $scriptRoot $SvgName)) {
        $IconDir = $scriptRoot
    } else {
        $IconDir = (Get-Location).Path
    }
}

$IconDir = (Resolve-Path -LiteralPath $IconDir).Path
$svgPath = Join-Path $IconDir $SvgName

if (-not (Test-Path -LiteralPath $svgPath)) {
    throw "Source SVG not found: $svgPath`nPass -IconDir <path-to-folder-containing-$SvgName>."
}

Write-Host "Icon directory : $IconDir"
Write-Host "Source SVG     : $svgPath"
Write-Host "Sizes          : $($Sizes -join ', ')"
Write-Host ''

# ---------------------------------------------------------------------------
# Detect an SVG rasterizer.
# ---------------------------------------------------------------------------
function Find-Tool {
    param([string[]]$Names)
    foreach ($n in $Names) {
        $cmd = Get-Command $n -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

$rsvg     = Find-Tool @('rsvg-convert', 'rsvg-convert.exe')
$inkscape = if (-not $rsvg) { Find-Tool @('inkscape', 'inkscape.exe') } else { $null }
$magick   = if (-not $rsvg -and -not $inkscape) { Find-Tool @('magick', 'magick.exe') } else { $null }

if     ($rsvg)     { $renderer = 'rsvg';     Write-Host "Renderer       : rsvg-convert ($rsvg)" }
elseif ($inkscape) { $renderer = 'inkscape'; Write-Host "Renderer       : Inkscape ($inkscape)" }
elseif ($magick)   { $renderer = 'magick';   Write-Host "Renderer       : ImageMagick ($magick)" }
else {
    throw @"
No SVG rasterizer found. Install one of:
  winget install --id GNOME.librsvg          # rsvg-convert (recommended)
  winget install --id Inkscape.Inkscape      # Inkscape
  winget install --id ImageMagick.ImageMagick# ImageMagick (magick)
Then re-run this script.
"@
}
Write-Host ''

# ---------------------------------------------------------------------------
# Render one PNG at a given size (transparent background, exact square).
# ---------------------------------------------------------------------------
function Convert-SvgToPng {
    param([int]$Size, [string]$OutPath)

    switch ($renderer) {
        'rsvg' {
            & $rsvg -w $Size -h $Size --background-color none `
                    $svgPath -o $OutPath
        }
        'inkscape' {
            # Inkscape 1.x CLI. --export-background-opacity=0 keeps transparency.
            & $inkscape $svgPath `
                    --export-type=png `
                    --export-filename=$OutPath `
                    --export-width=$Size `
                    --export-height=$Size `
                    --export-background-opacity=0 | Out-Null
        }
        'magick' {
            # -background none before the input keeps the canvas transparent.
            & $magick -background none $svgPath `
                    -resize "$($Size)x$($Size)" `
                    -define png:color-type=6 `
                    $OutPath
        }
    }

    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutPath)) {
        throw "Renderer failed to produce $OutPath (size $Size)."
    }
}

# ---------------------------------------------------------------------------
# Render all requested PNG sizes.
# ---------------------------------------------------------------------------
$pngPaths = [ordered]@{}      # size -> path
$createdByUs = @()            # track files we made, for -IcoOnly cleanup

foreach ($s in ($Sizes | Sort-Object -Unique)) {
    $out = Join-Path $IconDir ("icon{0}.png" -f $s)
    $existed = Test-Path -LiteralPath $out
    Write-Host ("Rendering {0,4}x{0,-4} -> {1}" -f $s, (Split-Path $out -Leaf))
    Convert-SvgToPng -Size $s -OutPath $out
    $pngPaths[$s] = $out
    if (-not $existed) { $createdByUs += $out }
}
Write-Host ''

# ---------------------------------------------------------------------------
# Pack the PNGs into a multi-resolution icon.ico (pure PowerShell).
#
# ICO layout:
#   ICONDIR    : reserved(0,u16) type(1,u16) count(u16)
#   ICONDIRENTRY x count :
#       width(u8) height(u8) colorCount(u8) reserved(u8)
#       planes(u16) bitCount(u16) bytesInRes(u32) imageOffset(u32)
#   ... then each image blob (here: the PNG file bytes, stored verbatim).
#   width/height byte = 0 means 256.
# ---------------------------------------------------------------------------
function Write-Ico {
    param(
        [System.Collections.Specialized.OrderedDictionary]$Pngs,  # size -> path
        [string]$OutPath
    )

    $sizes = @($Pngs.Keys | Sort-Object)
    $count = $sizes.Count

    # Read every PNG blob up front.
    $blobs = @{}
    foreach ($s in $sizes) {
        $blobs[$s] = [System.IO.File]::ReadAllBytes($Pngs[$s])
    }

    $fs = [System.IO.File]::Open($OutPath, [System.IO.FileMode]::Create,
                                 [System.IO.FileAccess]::Write)
    try {
        $bw = New-Object System.IO.BinaryWriter($fs)

        # ICONDIR
        $bw.Write([UInt16]0)          # reserved
        $bw.Write([UInt16]1)          # type = icon
        $bw.Write([UInt16]$count)     # image count

        # Offsets start after the header + all directory entries.
        $offset = 6 + (16 * $count)

        # ICONDIRENTRY table
        foreach ($s in $sizes) {
            $blob = $blobs[$s]
            $dim  = if ($s -ge 256) { 0 } else { $s }   # 0 encodes 256
            $bw.Write([Byte]$dim)         # width
            $bw.Write([Byte]$dim)         # height
            $bw.Write([Byte]0)            # color count (0 = >=256 or truecolor)
            $bw.Write([Byte]0)            # reserved
            $bw.Write([UInt16]1)          # color planes
            $bw.Write([UInt16]32)         # bits per pixel
            $bw.Write([UInt32]$blob.Length)   # size of image data
            $bw.Write([UInt32]$offset)        # offset to image data
            $offset += $blob.Length
        }

        # Image blobs, in the same order.
        foreach ($s in $sizes) {
            $bw.Write($blobs[$s])
        }

        $bw.Flush()
    }
    finally {
        $fs.Dispose()
    }
}

if (-not $PngOnly) {
    $icoPath = Join-Path $IconDir 'icon.ico'
    Write-Host ("Packing icon.ico ({0} sizes: {1}) -> {2}" -f `
        $pngPaths.Count, ($pngPaths.Keys -join ','), (Split-Path $icoPath -Leaf))
    Write-Ico -Pngs $pngPaths -OutPath $icoPath

    $icoLen = (Get-Item -LiteralPath $icoPath).Length
    Write-Host ("  icon.ico written: {0:N0} bytes" -f $icoLen)
    Write-Host ''
}

# ---------------------------------------------------------------------------
# -IcoOnly: remove PNGs this run created that weren't wanted as final output.
# ---------------------------------------------------------------------------
if ($IcoOnly -and $createdByUs.Count -gt 0) {
    Write-Host "IcoOnly: removing intermediate PNGs this run created..."
    foreach ($f in $createdByUs) {
        Remove-Item -LiteralPath $f -Force
        Write-Host "  removed $(Split-Path $f -Leaf)"
    }
    Write-Host ''
}

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
Write-Host "Done." -ForegroundColor Green
Write-Host "Output in: $IconDir"
if (-not $IcoOnly) {
    foreach ($s in ($pngPaths.Keys)) {
        Write-Host ("  icon{0}.png" -f $s)
    }
}
if (-not $PngOnly) {
    Write-Host "  icon.ico"
}
