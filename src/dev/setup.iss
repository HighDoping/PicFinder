#define MyAppName "PicFinder"
#define MyAppPathName "PicFinder"
#define MyAppPublisher "HighDoping"
#define MyAppExeName "main.exe"
#define MyAppPath "{#SourcePath}\main.dist"


[Setup]
AppId={{DFD81750-7230-4E43-ACAC-9830A32CD9A9}
AppName={#MyAppName}
AppVersion=GetStringFileInfo("{#SourcePath}..\main.dist\main.exe", 'ProductVersion')
AppPublisher={#MyAppPublisher}
AppCopyright=GetStringFileInfo("{#SourcePath}..\main.dist\main.exe", 'LegalCopyright')
DefaultDirName={autopf}\{#MyAppPathName}
DisableProgramGroupPage=yes
PrivilegesRequiredOverridesAllowed=dialog
OutputDir= {#SourcePath}\..\installer_dist 
OutputBaseFilename={#MyAppName} Installer
Compression=lzma2/ultra64
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
SolidCompression=yes
WizardStyle=modern
AllowNoIcons=yes
; PrivilegesRequired=admin
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked


[Files]
Source: "{#SourcePath}..\main.dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourcePath}..\main.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon


