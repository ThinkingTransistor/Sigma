/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A time step which defines a time dependency for every <see cref="Interval"/> times that <see cref="TimeScale"/> happens (and all that a total of <see cref="LiveTime"/> times).
	/// </summary>
	public interface ITimeStep
	{
		/// <summary>
		/// The time scale of this time step, i.e. the base unit (e.g. EPOCH or UPDATE).
		/// </summary>
		TimeScale TimeScale { get; }

		/// <summary>
		/// The interval between time steps in time scale units.
		/// </summary>
		int Interval { get; }

		/// <summary>
		/// The total number of times this time step should be active, or a negative number if it should always be active. 
		/// </summary>
		int LiveTime { get; }
	}

	/// <summary>
	/// The default implementation of the <see cref="ITimeStep"/> interface.
	/// Represents a time step which defines a time dependency for every <see cref="Interval"/> times that <see cref="TimeScale"/> happens (and all that a total of <see cref="LiveTime"/> times).
	/// </summary>
	public class TimeStep : ITimeStep
	{
		public const int LiveTimeForever = -1;

		public TimeScale TimeScale { get; internal set; }
		public int Interval { get; internal set; }
		public int LiveTime { get; internal set; }

		public TimeStep(TimeScale timeScale, int interval, int liveTime = LiveTimeForever)
		{
			if (timeScale == null)
			{
				throw new ArgumentNullException(nameof(timeScale));
			}

			if (interval < 0)
			{
				throw new ArgumentNullException($"Time step interval must be >= 1, but was {interval}.");
			}

			if (liveTime == 0)
			{
				throw new ArgumentException("Live time cannot be zero, but was, well, zero.");
			}

			TimeScale = timeScale;
			Interval = interval;
			LiveTime = liveTime;
		}
	}

	/// <summary>
	/// A time scale, the basic unit for <see cref="ITimeStep"/> (e.g. epoch or iteration).
	/// </summary>
	public class TimeScale
	{
		public static readonly TimeScale Epoch = new TimeScale();
		public static readonly TimeScale Iteration = new TimeScale();
	}
}
