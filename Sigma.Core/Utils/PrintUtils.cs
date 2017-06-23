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

		/// <summary>
		/// Format a millisecond time to a simple string, e.g. "12.423min". 
		/// </summary>
		/// <param name="timeMilliseconds">The time in milliseconds.</param>
		/// <param name="format">The time unit format.</param>
		/// <returns>The time string.</returns>
		public static string FormatTimeSimple(double timeMilliseconds, TimeUnitFormat format = TimeUnitFormat.Minimum)
		{
			return FormatTime(timeMilliseconds, separator: ".", format: format, depth: 1, putFirstUnitLast: true, padSubUnits: true);
		}

		/// <summary>
		/// Format a millisecond time to a string with a certain time unit and depth (e.g. "3min, 14sec, 15ms" with depth 3).
		/// </summary>
		/// <param name="timeMilliseconds">The time in milliseconds.</param>
		/// <param name="separator">The seperator between each unit (", " by default).</param>
		/// <param name="format">The time unit format.</param>
		/// <param name="depth">The depth to represent units with (depth 0 = just top unit, depth 1 = top unit and 1 specific unit, ...)</param>
		/// <param name="putFirstUnitLast">Indicate if no other unit than the first unit should be put in the string. First unit will be put at the end of the string (for e.g. "3.14min").</param>
		/// <param name="padSubUnits">Indicate if sub units should be left-padded with zeros </param>
		/// <returns>The time string.</returns>
		public static string FormatTime(double timeMilliseconds, string separator = ", ", TimeUnitFormat format = TimeUnitFormat.Minimum, int depth = 1, bool putFirstUnitLast = false, bool padSubUnits = false)
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

				if (!putFirstUnitLast) builder.Append(localUnit.GetTimeUnitInFormat(format));

				if (i - 1 >= printDepth)
				{
					builder.Append(separator);

					localUnit = localUnit.GetSmaller().Value;
				}
			}

			if (putFirstUnitLast) builder.Append(first.GetTimeUnitInFormat(format));

			return builder.ToString();
		}

		/// <summary>
		/// Get the inverse time for a certain time (e.g. 40ms for an iteration, how many iterations per what is that?).
		/// </summary>
		{
			resultUnit = TimeUnit.Millisecond;

			while (inverseTime < 1.0)
			{
				resultUnit = resultUnit.GetBigger().Value;
				inverseTime *= resultUnit.GetTimeUnitParts();
			}

			return inverseTime;
		}
	}
}
