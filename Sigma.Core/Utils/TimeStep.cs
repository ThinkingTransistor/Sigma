/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
		public const int LIVE_TIME_FOREVER = -1;

		public TimeScale TimeScale { get; internal set; }
		public int Interval { get; internal set; }
		public int LiveTime { get; internal set; }

		public TimeStep(TimeScale timeScale, int interval, int liveTime = LIVE_TIME_FOREVER)
		{
			if (timeScale == null)
			{
				throw new ArgumentNullException("Timescale cannot be null.");
			}

			if (interval < 0)
			{
				throw new ArgumentNullException($"Time step interval must be >= 1, but was {interval}.");
			}

			if (liveTime == 0)
			{
				throw new ArgumentException($"Live time cannot be zero, but was, well, zero.");
			}

			this.TimeScale = timeScale;
			this.Interval = interval;
			this.LiveTime = liveTime;
		}
	}

	/// <summary>
	/// A time scale, the basic unit for <see cref="ITimeStep"/> (e.g. EPOCH or UPDATE).
	/// </summary>
	public class TimeScale
	{
		public static readonly TimeScale EPOCH = new TimeScale();
		public static readonly TimeScale UPDATE = new TimeScale();
	}
}
