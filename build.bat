REM @echo Off
set config=%1
if "%config%" == "" (
   set config=Release
)
 
set version=1.0.0
if not "%PackageVersion%" == "" (
   set version=%PackageVersion%
)

set nuget=
if "%nuget%" == "" (
	set nuget=nuget
)

set build="%programfiles(x86)%\MSBuild\14.0\Bin\MsBuild.exe"

powershell -Command "(new-object System.Net.WebClient).DownloadFile('https://dist.nuget.org/win-x86-commandline/latest/nuget.exe','nuget.exe')"

copy nuget.exe Sigma.Core
copy nuget.exe Sigma.Core.Monitors.WPF

REM old: "%build%" Sigma.sln /p:Configuration="%config%" /p:Platform=x64

REM build Sigma.Core
cd Sigma.Core
	
	nuget.exe restore -SolutionDirectory ../

	%build% Sigma.Core.csproj /p:Configuration="%config%" /p:Platform=x64

	REM pack Sigma.Core

	nuget.exe pack "Sigma.Core.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%


REM build Sigma.Core.Monitors.WPF
cd ../Sigma.Core.Monitors.WPF

	REM update the targets file
	cd build
	powershell -Command "(gc Sigma.Core.Monitors.WPF.targets) -replace '~version~', '%version%' | Out-File Sigma.Core.Monitors.WPF.targets"

	REM actual build
	nuget.exe restore -SolutionDirectory ../

	%build% Sigma.Core.Monitors.WPF.csproj /p:Configuration="%config%" /p:Platform=x64
	
	REM pack Sigma.Core.Monitors.WPF
	cd ..
	nuget.exe pack "Sigma.Core.Monitors.WPF.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%