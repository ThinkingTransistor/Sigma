/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	public static class SystemInformationUtils
	{
		private static PerformanceCounter memoryCounterAvailableKBytes;

		static SystemInformationUtils()
		{
			memoryCounterAvailableKBytes = new PerformanceCounter("Memory", "Available KBytes");
		}

		public static long GetAvailablePhysicalMemory()
		{
			return (long) memoryCounterAvailableKBytes.NextValue() * 1024L;
		}
	}
}
