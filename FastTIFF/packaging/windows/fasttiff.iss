; Inno Setup script for the FastTIFF Windows installer.
;
; Built in CI (the build-windows job in .github/workflows/release.yml) with:
;     iscc /DMyAppVersion=<x.y.z> FastTIFF\packaging\windows\fasttiff.iss
; It packages the already-built FastTIFF.exe (the wgpu/DX12 release the job
; stages at the repo root) and emits FastTIFF-<version>-setup.exe there.
;
; Per-user install (no UAC): `PrivilegesRequired=lowest` puts the app under the
; user's Programs folder and registers file handlers in HKCU. Paths below are
; relative to THIS script's directory (Inno's default SourceDir), so the repo
; layout — not the CI working directory — is what they resolve against:
;   ..\..           -> FastTIFF\      ..\..\..        -> repo root

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#define MyAppName "Fast TIFF Viewer"
#define MyAppExeName "FastTIFF.exe"
#define MyAppPublisher "konung_yaropolk"
#define MyAppURL "https://github.com/konung-yaropolk/FastTIFF"

[Setup]
; AppId uniquely identifies the app for upgrades/uninstall — never change it
; once released, or upgrades install side-by-side instead of replacing.
AppId={{8F3B2A14-9C7D-4E6A-B1F0-2D5E8C9A7B34}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\FastTIFF
DefaultGroupName=FastTIFF
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
; 64-bit only (the app has no 32-bit build).
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Per-user, no administrator prompt.
PrivilegesRequired=lowest
OutputDir=..\..\..
OutputBaseFilename=FastTIFF-{#MyAppVersion}-setup
SetupIconFile=..\..\icon\icon.ico
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc"; Description: "Add FastTIFF to the ""Open with"" list for .tif and .tiff files"; GroupDescription: "File associations:"

[Files]
Source: "..\..\..\FastTIFF.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\FastTIFF"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,FastTIFF}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FastTIFF"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Register FastTIFF as an *application* that handles TIFFs so it appears in
; Explorer's right-click "Open with" list for .tif/.tiff. The Applications key
; with an open command plus a SupportedTypes list is the canonical mechanism
; Windows uses to populate "Open with". This does NOT seize the default handler
; — Windows 10+ reserves default-app changes for the user's own choice
; (Settings / "Open with > Always"). Everything lands in HKCU (per-user install)
; and the whole key tree is removed on uninstall (uninsdeletekey on the root).
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAppExeName}"; ValueType: none; Flags: uninsdeletekey; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAppExeName}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".tif"; ValueData: ""; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".tiff"; ValueData: ""; Tasks: assoc

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,FastTIFF}"; Flags: nowait postinstall skipifsilent
