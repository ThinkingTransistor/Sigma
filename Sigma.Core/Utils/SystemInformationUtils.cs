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
		public static long DefaultSystemMemoryAvailableBytes { get; set; } = 4L * 1024L * 1024L * 1024L; //4GB

		private static PerformanceCounter memoryCounterAvailableKBytes;

		static SystemInformationUtils()
		{
			memoryCounterAvailableKBytes = new PerformanceCounter("Memory", "Available KBytes");
		}

		public static long GetAvailablePhysicalMemoryBytes()
		{
			float readKBytes = memoryCounterAvailableKBytes.NextValue();
			long availableMemoryBytes = (long) readKBytes * 1024L;

			if (readKBytes <= 0)
			{
				availableMemoryBytes = DefaultSystemMemoryAvailableBytes;
			}

			return availableMemoryBytes;
		}
	}
}
