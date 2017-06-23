/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Globalization;
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
		public static readonly char[] Utf16GreyscalePalette = { ' ', '·', '-', '▴', '▪', '●', '♦', '■', '█' };

		/// <summary>
		/// An ASCII greyscale palette from min to max.
		/// </summary>
		public static readonly char[] AsciiGreyscalePalette = { ' ', '.', ':', 'x', 'T', 'Y', 'V', 'X', 'H', 'N', 'M' };

		/// <summary>
		/// All ASCII digits.
		/// </summary>
		public static readonly char[] AsciiDigits = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };

		public static string FormatTimeSimple(double timeMilliseconds, TimeUnitFormat format = TimeUnitFormat.Minimum)
		{
			return FormatTime(timeMilliseconds, separator: ".", format: format, depth: 1, printFirstUnitLast: true, padSubUnits: true);
		}

		public static string FormatTime(double timeMilliseconds, string separator = ", ", TimeUnitFormat format = TimeUnitFormat.Minimum, int depth = 1, bool printFirstUnitLast = false, bool padSubUnits = false)
		{
			TimeUnit localUnit = TimeUnit.Millisecond;

			IList<string> localTimes = new List<string>();
			double localTime = timeMilliseconds;

			while (true)
			{
				TimeUnit? nextUnit = localUnit.GetBigger();

				localTimes.Add(((int)localTime).ToString(CultureInfo.InvariantCulture));

				if (!nextUnit.HasValue) break;

				double timeUnitParts = nextUnit.Value.GetTimeUnitParts();
				double nextLocalTime = localTime / timeUnitParts;

				if (nextLocalTime < 1.0) break;

				string subTime = (localTime % timeUnitParts).ToString(CultureInfo.InvariantCulture);

				if (padSubUnits) subTime = subTime.PadLeft((int)Math.Log10(timeUnitParts), '0');

				localTimes[localTimes.Count - 1] = subTime;

				localTime /= nextUnit.Value.GetTimeUnitParts();
				localUnit = nextUnit.Value;
			}

			StringBuilder builder = new StringBuilder();

			TimeUnit first = localUnit;

			int printDepth = Math.Max(0, localTimes.Count - depth - 1);
			for (int i = localTimes.Count - 1; i >= printDepth; i--)
			{
				builder.Append(localTimes[i]);

				if (!printFirstUnitLast) builder.Append(localUnit.GetTimeUnitInFormat(format));

				if (i - 1 >= printDepth)
				{
					builder.Append(separator);

					localUnit = localUnit.GetSmaller().Value;
				}
			}

			if (printFirstUnitLast) builder.Append(first.GetTimeUnitInFormat(format));

			return builder.ToString();
		}
	}
}
