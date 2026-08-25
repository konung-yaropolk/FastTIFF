@echo off
setlocal EnableDelayedExpansion
REM ===========================================================================
REM  generate-icons.bat
REM
REM  Renders every SVG in the target directory into all the PNG sizes the build
REM  needs, then packs a multi-resolution .ico from those PNGs for each file.
REM  Run this ONCE at dev time (and again whenever SVGs change) and commit 
REM  the results.
REM
REM  Why Inkscape: SVG meshes require Inkscape for correct color rendering.
REM
REM  Requirements:
REM    - Inkscape 1.x on PATH (or set INKSCAPE below to its full path).
REM        winget install --id Inkscape.Inkscape
REM    - PowerShell (built into Windows) -- used only to assemble .ico files.
REM
REM  Output (written next to the source SVGs):
REM    <name>16.png <name>32.png <name>48.png <name>64.png <name>128.png 
REM    <name>256.png <name>512.png <name>1024.png
REM    <name>.ico (multi-res: 16,32,48,64,128,256)
REM ===========================================================================

REM --- Locate the icon directory (this script's folder, then \FastTIFF\icon,
REM     else assume the script sits in the icon dir itself). --------------------
set "SCRIPT_DIR=%~dp0"
set "ICON_DIR=%SCRIPT_DIR%FastTIFF\icon"
if not exist "%ICON_DIR%\*.svg" (
    if exist "%SCRIPT_DIR%*.svg" (
        set "ICON_DIR=%SCRIPT_DIR%"
    )
)
REM Strip any trailing backslash for consistency.
if "%ICON_DIR:~-1%"=="\" set "ICON_DIR=%ICON_DIR:~0,-1%"

if not exist "%ICON_DIR%\*.svg" (
    echo [ERROR] No .svg files found in "%ICON_DIR%".
    echo         Run this script from the repo root, or place it in FastTIFF\icon\.
    exit /b 1
)

REM --- Find Inkscape. Prefer PATH; fall back to common install locations. -----
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
echo Inkscape : %INKSCAPE%

set "SIZES=16 32 48 64 128 256 512 1024"

REM --- Iterate through all SVGs in the directory -----------------------------
for %%F in ("%ICON_DIR%\*.svg") do (
    set "SVG=%%F"
    set "BASENAME=%%~nF"
    
    echo.
    echo =======================================================================
    echo Processing: !BASENAME!.svg
    echo =======================================================================
    
    REM --- Render each PNG size with Inkscape ---
    for %%S in (%SIZES%) do (
        set "OUT=%ICON_DIR%\!BASENAME!%%S.png"
        echo Rendering %%Sx%%S -^> !BASENAME!%%S.png
        "%INKSCAPE%" "!SVG!" ^
            --export-type=png ^
            --export-filename="!OUT!" ^
            --export-width=%%S ^
            --export-height=%%S ^
            --export-background-opacity=0 >nul 2>&1
            
        if not exist "!OUT!" (
            echo [ERROR] Inkscape failed to produce !BASENAME!%%S.png
            exit /b 1
        )
    )

    REM --- Pack .ico from the small PNGs, via built-in PowerShell ---
    echo Packing !BASENAME!.ico ^(16,32,48,64,128,256^)...
    
    set "ICO=%ICON_DIR%\!BASENAME!.ico"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $d='%ICON_DIR%'; $base='!BASENAME!'; $sizes=16,32,48,64,128,256; $blobs=@{}; foreach($s in $sizes){ $p=Join-Path $d ('{0}{1}.png' -f $base, $s); if(-not (Test-Path -LiteralPath $p)){ throw ('missing ' + $p) }; $blobs[$s]=[System.IO.File]::ReadAllBytes($p) }; $fs=[System.IO.File]::Open((Join-Path $d ('{0}.ico' -f $base)),[System.IO.FileMode]::Create,[System.IO.FileAccess]::Write); $bw=New-Object System.IO.BinaryWriter($fs); $bw.Write([UInt16]0); $bw.Write([UInt16]1); $bw.Write([UInt16]$sizes.Count); $off=6+16*$sizes.Count; foreach($s in $sizes){ $b=$blobs[$s]; if($s -ge 256){$dim=0}else{$dim=$s}; $bw.Write([Byte]$dim); $bw.Write([Byte]$dim); $bw.Write([Byte]0); $bw.Write([Byte]0); $bw.Write([UInt16]1); $bw.Write([UInt16]32); $bw.Write([UInt32]$b.Length); $bw.Write([UInt32]$off); $off+=$b.Length }; foreach($s in $sizes){ $bw.Write($blobs[$s]) }; $bw.Flush(); $fs.Dispose(); Write-Host ('  {0}.ico written: {1:N0} bytes' -f $base, (Get-Item -LiteralPath (Join-Path $d ('{0}.ico' -f $base))).Length)"
    
    if errorlevel 1 (
        echo [ERROR] Failed to build !BASENAME!.ico
        exit /b 1
    )
)

echo.
echo =======================================================================
echo Done. Generated PNGs and ICOs for all SVGs in %ICON_DIR%.
echo Commit these files. The CI workflows consume them directly.
endlocal