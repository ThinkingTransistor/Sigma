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

dir

echo "========"
echo %WINDIR%

nuget restore Sigma.sln

%WINDIR%\Microsoft.NET\Framework\v4.0.30319\msbuild Sigma.sln /p:Configuration="%config%" /m /v:M /fl /flp:LogFile=msbuild.log;Verbosity=diag /nr:false

REM pack Sigma.Core

cd Sigma.Core
nuget pack "Sigma.Core.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%

cd ..

REM pack Sigma.Core.Monitors.WPF

cd Sigma.Core.Monitors.WPF

REM update the targets file

cd build
powershell -Command "(gc Sigma.Core.Monitors.WPF.targets) -replace '~version~', '%version%' | Out-File Sigma.Core.Monitors.WPF.targets"

cd ..
nuget pack "Sigma.Core.Monitors.WPF.csproj" -IncludeReferencedProjects -Prop Platform=x64 -Verbosity detailed -Prop Configuration=%config% -Version %version%