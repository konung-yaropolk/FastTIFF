@echo off
setlocal EnableDelayedExpansion
REM ===========================================================================
REM  generate-icons.bat
REM
REM  Renders FastTIFF\icon\icon.svg into every PNG size the build needs, then
REM  packs a multi-resolution icon.ico from those PNGs. Run this ONCE at dev
REM  time (and again whenever icon.svg changes) and commit the results.
REM
REM  Why Inkscape: icon.svg uses an SVG mesh gradient, which headless CLI
REM  rasterizers (librsvg / ImageMagick) cannot render -- they'd output a
REM  gray icon. Inkscape has a native mesh-gradient renderer, so it produces
REM  the correct colors. The CI workflows consume these committed PNGs/ICO;
REM  they do NOT rasterize the SVG themselves.
REM
REM  Requirements:
REM    - Inkscape 1.x on PATH (or set INKSCAPE below to its full path).
REM        winget install --id Inkscape.Inkscape
REM    - PowerShell (built into Windows) -- used only to assemble icon.ico.
REM
REM  Output (written next to icon.svg, in FastTIFF\icon\):
REM    icon16.png icon32.png icon48.png icon64.png icon128.png icon256.png
REM    icon512.png icon1024.png        (512/1024 feed the macOS .icns build)
REM    icon.ico                        (multi-res: 16,32,48,64,128,256)
REM ===========================================================================

REM --- Locate the icon directory (this script's folder, then \FastTIFF\icon,
REM     else assume the script sits in the icon dir itself). --------------------
set "SCRIPT_DIR=%~dp0"
set "ICON_DIR=%SCRIPT_DIR%FastTIFF\icon"
if not exist "%ICON_DIR%\icon.svg" (
    if exist "%SCRIPT_DIR%icon.svg" (
        set "ICON_DIR=%SCRIPT_DIR%"
    )
)
REM Strip any trailing backslash for consistency.
if "%ICON_DIR:~-1%"=="\" set "ICON_DIR=%ICON_DIR:~0,-1%"

set "SVG=%ICON_DIR%\icon.svg"
if not exist "%SVG%" (
    echo [ERROR] icon.svg not found at "%SVG%".
    echo         Run this script from the repo root, or place it in FastTIFF\icon\.
    exit /b 1
)

REM --- Find Inkscape. Prefer PATH; fall back to common install locations. -----
REM  NOTE: %ProgramFiles(x86)% contains a ")" which would prematurely close a
REM  parenthesized IF block, so each candidate is checked on its own single line
REM  (no ( ) blocks wrapping the (x86) variable).
set "INKSCAPE="
where inkscape >nul 2>nul && set "INKSCAPE=inkscape"
if not defined INKSCAPE if exist "%ProgramFiles%\Inkscape\bin\inkscape.exe" set "INKSCAPE=%ProgramFiles%\Inkscape\bin\inkscape.exe"
if not defined INKSCAPE if exist "%ProgramFiles%\Inkscape\inkscape.exe" set "INKSCAPE=%ProgramFiles%\Inkscape\inkscape.exe"
if not defined INKSCAPE if exist "%ProgramFiles(x86)%\Inkscape\bin\inkscape.exe" set "INKSCAPE=%ProgramFiles(x86)%\Inkscape\bin\inkscape.exe"
if not defined INKSCAPE if exist "%ProgramFiles(x86)%\Inkscape\inkscape.exe" set "INKSCAPE=%ProgramFiles(x86)%\Inkscape\inkscape.exe"
if not defined INKSCAPE goto :no_inkscape
goto :have_inkscape

:no_inkscape
echo [ERROR] Inkscape not found on PATH or in Program Files.
echo         Install it:  winget install --id Inkscape.Inkscape
echo         Or edit this script and set INKSCAPE to inkscape.exe's full path.
exit /b 1

:have_inkscape

echo Icon dir : %ICON_DIR%
echo Source   : %SVG%
echo Inkscape : %INKSCAPE%
echo.

REM --- Render each PNG size with Inkscape. -----------------------------------
REM  --export-background-opacity=0 keeps the canvas transparent.
REM  A square SVG viewport means width==height keeps the aspect ratio.
set "SIZES=16 32 48 64 128 256 512 1024"

for %%S in (%SIZES%) do (
    set "OUT=%ICON_DIR%\icon%%S.png"
    echo Rendering %%Sx%%S -^> icon%%S.png
    "%INKSCAPE%" "%SVG%" ^
        --export-type=png ^
        --export-filename="!OUT!" ^
        --export-width=%%S ^
        --export-height=%%S ^
        --export-background-opacity=0 >nul 2>&1
    if not exist "!OUT!" (
        echo [ERROR] Inkscape failed to produce icon%%S.png
        exit /b 1
    )
)
echo.

REM --- Pack icon.ico from the small PNGs, via built-in PowerShell. ------------
REM  Windows reads PNG-compressed ICO frames (Vista+), so each size is embedded
REM  as its PNG bytes. The inline PS writes the ICONDIR + ICONDIRENTRY table
REM  then the blobs. ICO frames: 16,32,48,64,128,256 (256 stored as PNG).
echo Packing icon.ico (16,32,48,64,128,256)...

REM  The PS command is on ONE physical line and uses only single quotes inside,
REM  so there are no caret-continuations or \" escapes for the batch parser to
REM  mangle. %ICON_DIR% is expanded by batch into the $d variable up front.
set "ICO=%ICON_DIR%\icon.ico"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $d='%ICON_DIR%'; $sizes=16,32,48,64,128,256; $blobs=@{}; foreach($s in $sizes){ $p=Join-Path $d ('icon{0}.png' -f $s); if(-not (Test-Path -LiteralPath $p)){ throw ('missing ' + $p) }; $blobs[$s]=[System.IO.File]::ReadAllBytes($p) }; $fs=[System.IO.File]::Open((Join-Path $d 'icon.ico'),[System.IO.FileMode]::Create,[System.IO.FileAccess]::Write); $bw=New-Object System.IO.BinaryWriter($fs); $bw.Write([UInt16]0); $bw.Write([UInt16]1); $bw.Write([UInt16]$sizes.Count); $off=6+16*$sizes.Count; foreach($s in $sizes){ $b=$blobs[$s]; if($s -ge 256){$dim=0}else{$dim=$s}; $bw.Write([Byte]$dim); $bw.Write([Byte]$dim); $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([UInt16]1); $bw.Write([UInt16]32); $bw.Write([UInt32]$b.Length); $bw.Write([UInt32]$off); $off+=$b.Length }; foreach($s in $sizes){ $bw.Write($blobs[$s]) }; $bw.Flush(); $fs.Dispose(); Write-Host ('  icon.ico written: {0:N0} bytes' -f (Get-Item -LiteralPath (Join-Path $d 'icon.ico')).Length)"

if errorlevel 1 (
    echo [ERROR] Failed to build icon.ico
    exit /b 1
)

echo.
echo Done. Generated in %ICON_DIR%:
for %%S in (%SIZES%) do echo   icon%%S.png
echo   icon.ico
echo.
echo Commit these files. The CI workflows consume them directly.
endlocal
