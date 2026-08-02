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
; Stable (version-less) filename so the README can link to it via GitHub's
; releases/latest/download/ redirect. The version still shows in Add/Remove
; Programs (AppVersion above).
OutputBaseFilename=FastTIFF-setup
SetupIconFile=..\..\icon\icon.ico
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
; The "assoc" task writes file-type keys below; this makes Setup notify the
; shell (SHChangeNotify) so Explorer refreshes TIFF icons without a re-login.
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc"; Description: "Associate .tif / .tiff with FastTIFF (adds it to ""Open with"" and gives those files FastTIFF's icon once it's your default TIFF app)"; GroupDescription: "File associations:"

[Files]
Source: "..\..\..\FastTIFF.exe"; DestDir: "{app}"; Flags: ignoreversion
; Bundle the icon into the install folder so the file-type association can point
; DefaultIcon at a standalone .ico (rather than the exe's embedded resource).
Source: "..\..\icon\icon.ico"; DestDir: "{app}"; DestName: "FastTIFF.ico"; Flags: ignoreversion

[Icons]
Name: "{group}\FastTIFF"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,FastTIFF}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\FastTIFF"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Register a ProgID (FastTIFF.tiff) for FastTIFF's TIFF handling: its
; DefaultIcon gives .tif/.tiff files FastTIFF's icon, and its open command is
; how it launches them. Listing that ProgID under each extension's
; OpenWithProgids adds FastTIFF to Explorer's "Open with" list *without* seizing
; the default handler — Windows 10+ reserves default-app changes for the user's
; own choice (Settings / "Open with > Always"), and the file icon only actually
; changes once FastTIFF is that default. All per-user (HKCU) and removed on
; uninstall: uninsdeletekey drops the whole ProgID tree, uninsdeletevalue removes
; just our entries from the system-owned .tif/.tiff keys.
Root: HKCU; Subkey: "Software\Classes\FastTIFF.tiff"; ValueType: string; ValueName: ""; ValueData: "TIFF Image"; Flags: uninsdeletekey; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\FastTIFF.tiff\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\FastTIFF.ico"; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\FastTIFF.tiff\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\.tif\OpenWithProgids"; ValueType: string; ValueName: "FastTIFF.tiff"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc
Root: HKCU; Subkey: "Software\Classes\.tiff\OpenWithProgids"; ValueType: string; ValueName: "FastTIFF.tiff"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,FastTIFF}"; Flags: nowait postinstall skipifsilent
