/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// Time unit utilities for getting information about time units.
	/// </summary>
	public static class TimeUnitExtensions
	{
		private static readonly IList<TimeUnit> AllTimeUnits = (TimeUnit[]) Enum.GetValues(typeof(TimeUnit));

		/// <summary>
		/// Get the number of sub-parts for a certain time unit (e.g. 12 months for a given year).
		/// </summary>
		/// <param name="unit">The unit.</param>
		/// <returns>A number of sub-parts for a given time unit.</returns>
		public static int GetTimeUnitParts(this TimeUnit unit)
		{
			switch (unit)
			{
				case TimeUnit.Year:
					return 12;
				case TimeUnit.Month:
					return 4;
				case TimeUnit.Week:
					return 7;
				case TimeUnit.Day:
					return 24;
				case TimeUnit.Hour:
					return 60;
				case TimeUnit.Minute:
					return 60;
				case TimeUnit.Second:
					return 1000;
				case TimeUnit.Millisecond:
					return 1;
				default:
					throw new ArgumentException($"TimeUnit {unit} is not a valid time unit.");
			}
		}

		/// <summary>
		/// Get a time unit name in a certain format.
		/// </summary>
		/// <param name="unit">The time unit.</param>
		/// <param name="format">The format (full or minimum).</param>
		/// <returns>The time unit name in the given format.</returns>
		public static string GetTimeUnitInFormat(this TimeUnit unit, TimeUnitFormat format)
		{
			switch (unit)
			{
				case TimeUnit.Year:
					return format == TimeUnitFormat.Full ? " years" : "y";
				case TimeUnit.Month:
					return format == TimeUnitFormat.Full ? " months" : "M";
				case TimeUnit.Day:
					return format == TimeUnitFormat.Full ? " days" : "d";
				case TimeUnit.Hour:
					return format == TimeUnitFormat.Full ? " hours" : "h";
				case TimeUnit.Minute:
					return format == TimeUnitFormat.Full ? " minutes" : "m";
				case TimeUnit.Second:
					return format == TimeUnitFormat.Full ? " seconds" : "s";
				case TimeUnit.Millisecond:
					return format == TimeUnitFormat.Full ? " milliseconds" : "ms";
				default:
					throw new ArgumentException($"TimeUnit {unit} is not a valid time unit.");
			}
		}


		/// <summary>
		/// Get the next biggest time unit for a certain unit.
		/// </summary>
		/// <param name="unit">The time unit.</param>
		/// <returns>The next biggest time unit.</returns>
		public static TimeUnit? GetBigger(this TimeUnit unit)
		{
			int index = AllTimeUnits.IndexOf(unit);

			if (index > 0 && index < AllTimeUnits.Count)
			{
				return AllTimeUnits[index - 1];
			}

			return null;
		}

		/// <summary>
		/// Get the next smallest time unit for a certain unit.
		/// </summary>
		/// <param name="unit">The time unit.</param>
		/// <returns>The next smallest time unit.</returns>
		public static TimeUnit? GetSmaller(this TimeUnit unit)
		{
			int index = AllTimeUnits.IndexOf(unit);

			if (index >= 0 && index < AllTimeUnits.Count - 1)
			{
				return AllTimeUnits[index + 1];
			}

			return null;
		}
	}

	/// <summary>
	/// A time unit (e.g. year, month, hour, second).
	/// </summary>
	public enum TimeUnit
	{
		/// <summary>
		/// A year, consisting of 12 months.
		/// </summary>
		Year,

		/// <summary>
		/// A month, consisting of 4 weeks (close enough).
		/// </summary>
		Month,

		/// <summary>
		/// A week, consisting of 7 days.
		/// </summary>
		Week,

		/// <summary>
		/// A day, consisting of 2 hours.
		/// </summary>
		Day,

		/// <summary>
		/// An hour, consisting of 60 minutes.
		/// </summary>
		Hour,

		/// <summary>
		/// A minute, consisting of 60 seconds.
		/// </summary>
		Minute,

		/// <summary>
		/// A second, consisting of 1000 milliseconds.
		/// </summary>
		Second,

		/// <summary>
		/// A millisecond, consisting of ... 1 millisecond.
		/// </summary>
		Millisecond
	}

	/// <summary>
	/// A time unit format (for time unit names).
	/// </summary>
	public enum TimeUnitFormat
	{
		/// <summary>
		/// A full format (e.g. year, minute).
		/// </summary>
		Full,

		/// <summary>
		/// A minimal format (e.g. yr, min, sec).
		/// </summary>
		Minimum,
	}
}
