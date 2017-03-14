/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Text;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility functions and constants for pretty printing various things to console (string formatting).
	/// </summary>
	public static class PrintUtils
	{
		/// <summary>
		/// A UTF-16 greyscale palette from min to max.
		/// </summary>
		public static readonly char[] Utf16GreyscalePalette = {' ', '·', '-', '▴', '▪', '●', '♦', '■', '█'};

		/// <summary>
		/// An ASCII greyscale palette from min to max.
		/// </summary>
		public static readonly char[] AsciiGreyscalePalette = { ' ', '.', ':', 'x', 'T', 'Y', 'V', 'X', 'H', 'N', 'M' };

		public static string GetFormattedTime(double timeMilliseconds, string separator = ", ", TimeUnitFormat format = TimeUnitFormat.Minimum, int depth = 1)
		{
			TimeUnit localUnit = TimeUnit.Millisecond;

			IList<double> localTimes = new List<double>();
			double localTime = timeMilliseconds;

			while (true)
			{
				TimeUnit? nextUnit = localUnit.GetBigger();

				localTimes.Add(localTime);

				if (!nextUnit.HasValue)
				{
					break;
				}

				double timeUnitParts = nextUnit.Value.GetTimeUnitParts();
				double nextLocalTime = localTime / timeUnitParts;
				if (nextLocalTime < 1.0)
				{
					break;
				}

				localTimes[localTimes.Count - 1] = localTime % timeUnitParts;

				localTime /= nextUnit.Value.GetTimeUnitParts();
				localUnit = nextUnit.Value;
			}

			StringBuilder builder = new StringBuilder();

			int printDepth = Math.Max(0, localTimes.Count - depth - 1);
			for (int i = localTimes.Count - 1; i >= printDepth; i--)
			{
				builder.Append((int) localTimes[i]);
				builder.Append(localUnit.GetTimeUnitInFormat(format));

				if (i - 1 >= printDepth)
				{
					builder.Append(separator);

					localUnit = localUnit.GetSmaller().Value;
				}
			}

			return builder.ToString();
		}
	}
}
