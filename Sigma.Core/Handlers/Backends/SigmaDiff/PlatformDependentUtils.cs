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
	internal static class PlatformDependentUtils
	{
		private static readonly ILog clazzLogger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private static bool _checkedPlatformDependentLibraries;

		internal static void CheckPlatformDependentLibraries()
		{
			if (_checkedPlatformDependentLibraries)
			{
				return;
			}

			PlatformID pid = Environment.OSVersion.Platform;

			if (pid == PlatformID.Win32NT || pid == PlatformID.Win32S || pid == PlatformID.Win32Windows || pid == PlatformID.WinCE)
			{
				clazzLogger.Info("Detected Windows system, using default windows DLLs (.dll).");
			}
			else if (pid == PlatformID.Unix)
			{
				clazzLogger.Info("Detected Unix systemdetected, using linux shared object files (.so).");
			}
			else if (pid == PlatformID.MacOSX)
			{
				clazzLogger.Info("Detected MacOSX system, using linux shared object files (.so).");
			}
			else if (pid == PlatformID.Xbox)
			{
				clazzLogger.Warn("Detected XBOX system. An XBOX. Really? I'm not even mad. Letting CLR decide if libraries work.");
			}
			else
			{
				clazzLogger.Warn($"Detected not natively supported system with platform id {pid} (potato system?). Letting CLR decide if libraries work.");
			}

			_checkedPlatformDependentLibraries = true;
		}
	}
}
