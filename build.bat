REM @echo Off
set config=%1
if "%config%" == "" (
   set config=NugetBuild
)
 
set version=1.0.0
if not "%PackageVersion%" == "" (
   set version=%PackageVersion%
)

set nuget=
if "%nuget%" == "" (
	set nuget=nuget
)

dir

echo "========"
echo %WINDIR%

powershell -Command "(new-object System.Net.WebClient).DownloadFile('https://dist.nuget.org/win-x86-commandline/latest/nuget.exe','nuget.exe')"

nuget.exe restore Sigma.sln


%programfiles(x86)%\MSBuild\14.0\Bin\MsBuild.exe Sigma.sln /p:Configuration="%config%" /p:Platform=x64

REM pack Sigma.Core

cd Sigma.Core
nuget.exe pack "Sigma.Core.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%

cd ..

REM pack Sigma.Core.Monitors.WPF

cd Sigma.Core.Monitors.WPF

REM update the targets file

cd build
powershell -Command "(gc Sigma.Core.Monitors.WPF.targets) -replace '~version~', '%version%' | Out-File Sigma.Core.Monitors.WPF.targets"

cd ..
nuget.exe pack "Sigma.Core.Monitors.WPF.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%