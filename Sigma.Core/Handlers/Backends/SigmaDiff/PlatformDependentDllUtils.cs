/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using System.Runtime.InteropServices;
using log4net;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// Utilities to handle platform dependent DLLs.
	/// </summary>
	internal static class PlatformDependentDllUtils
	{
		private static readonly ILog clazzLogger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private static bool _setPlatformDependentDllDirectory;

		internal static void EnsureSetPlatformDependentDllDirectory()
		{
			if (_setPlatformDependentDllDirectory)
			{
				return;
			}

			PlatformID pid = Environment.OSVersion.Platform;
			string dllSubDirectory;

			if (pid == PlatformID.Win32NT || pid == PlatformID.Win32S || pid == PlatformID.Win32Windows || pid == PlatformID.WinCE)
			{
				clazzLogger.Info("Windows system detected, setting platform dependent DLL sub-directory to Windows64.");

				dllSubDirectory = "Windows64";
			}
			else if (pid == PlatformID.Unix)
			{
				clazzLogger.Info("Unix system detected, setting plaform DLL sub-directory to Linux64.");

				dllSubDirectory = "Linux64";
			}
			else if (pid == PlatformID.MacOSX)
			{
				throw new NotSupportedException("MacOSX system detected, MacOSX DLL not supported as of now.");
			}
			else if (pid == PlatformID.Xbox)
			{
				throw new NotSupportedException("XBOX system detected. An XBOX. Really? I'm not even mad.");
			}
			else
			{
				throw new NotSupportedException($"Unsupported system with platform id {pid} (potato system?).");
			}

			string basePath = System.AppDomain.CurrentDomain.BaseDirectory + "Dependencies";
			string fullPath = Path.Combine(basePath, dllSubDirectory);

			SetDllDirectory(fullPath);

			clazzLogger.Info($"Set platform dependent DLL directory to \"{fullPath}\"");

			_setPlatformDependentDllDirectory = true;
		}

		[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
		private static extern bool SetDllDirectory(string path);
	}
}
