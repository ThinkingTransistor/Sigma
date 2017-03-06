using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// This utils-class provides the ability to check the process, OS.
	/// e.g. is 64 bit?
	/// </summary>
	public static class ProcessUtils
	{
		/// <summary>
		/// Determine whether the current application is running in 64bit mode.
		/// </summary>
		/// <returns><c>True</c> if the process is started 64 bit, <c>False</c> otherwise.</returns>
		public static bool Is64BitProcess()
		{
			return IntPtr.Size == 8;
		}

		/// <summary>
		/// Determines whether the current OS is 64 bit or not. 
		/// </summary>
		/// <returns><c>True</c> if the OS is 64 bit, <c>False</c> otherwise.</returns>
		public static bool Is64BitOs()
		{
			return Is64BitProcess() || InternalCheckIsWow64();
		}

		[DllImport("kernel32.dll", SetLastError = true, CallingConvention = CallingConvention.Winapi)]
		[return: MarshalAs(UnmanagedType.Bool)]
		private static extern bool IsWow64Process
		(
			[In] IntPtr hProcess,
			[Out] out bool wow64Process
		);

		/// <summary>
		/// Check if a process is running in 64bit mode -> 64bit OS.
		/// </summary>
		/// <returns><c>True</c> if any process runs in 64bit mode (and OS version is high enough). <c>False</c> otherwise.</returns>
		private static bool InternalCheckIsWow64()
		{
			if ((Environment.OSVersion.Version.Major == 5 && Environment.OSVersion.Version.Minor >= 1) ||
				Environment.OSVersion.Version.Major >= 6)
			{
				using (Process p = Process.GetCurrentProcess())
				{
					bool retVal;
					return IsWow64Process(p.Handle, out retVal) && retVal;
				}
			}
			return false;
		}
	}
}