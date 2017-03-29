/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// Utilities to handle platform dependent DLLs.
	/// </summary>
	internal static class PlatformDependentUtils
	{
		private static readonly ILog ClazzLogger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

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
				ClazzLogger.Debug("Detected Windows system, using windows libraries (.dll).");
			}
			else if (pid == PlatformID.Unix)
			{
				ClazzLogger.Debug("Detected Unix systemdetected, using linux libraries (.so).");
			}
			else if (pid == PlatformID.MacOSX)
			{
				ClazzLogger.Debug("Detected MacOSX system, using linux libraries (.so).");
			}
			else if (pid == PlatformID.Xbox)
			{
				ClazzLogger.Warn("Detected XBOX system. An XBOX. Really? I'm not even mad. Letting CLR decide what libraries to load.");
			}
			else
			{
				ClazzLogger.Warn($"Detected not natively supported system with platform id {pid} (potato system?). Letting CLR decide what libraries to load.");
			}

			_checkedPlatformDependentLibraries = true;
		}
	}
}
