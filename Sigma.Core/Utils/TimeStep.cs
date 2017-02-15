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
	public interface ITimeStep : IDeepCopyable, IComparable
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
		/// The local interval, counting down a local time step interval until next invocation.
		/// </summary>
		int LocalInterval { get; set; }

		/// <summary>
		/// The total number of times this time step should be active, or a negative number if it should always be active. 
		/// </summary>
		int LiveTime { get; }

		/// <summary>
		/// The local live time, counting down the local time to live.
		/// </summary>
		int LocalLiveTime { get; set; }

		/// <summary>
		/// Check if this time step equals another time steps time step (ignoring local time interval values).
		/// </summary>
		/// <param name="other">The other time step.</param>
		/// <returns>A boolean indicating if this time step equals another time steps time step.</returns>
		bool StepEquals(ITimeStep other);
	}

	/// <summary>
	/// The default implementation of the <see cref="ITimeStep"/> interface.
	/// Represents a time step which defines a time dependency for every <see cref="Interval"/> times that <see cref="TimeScale"/> happens (and all that a total of <see cref="LiveTime"/> times).
	/// </summary>
	public class TimeStep : ITimeStep
	{
		public const int LiveTimeForever = -1;

		public TimeScale TimeScale { get; }
		public int Interval { get; }
		public int LiveTime { get; }
		public int LocalInterval { get; set; }
		public int LocalLiveTime { get; set; }

		/// <summary>
		/// Create a time step with a certain time step interval, time step unit and live time.
		/// </summary>
		/// <param name="interval">The time step interval in <see cref="TimeScale"/> units.</param>
		/// <param name="timeScale">The time scale unit.</param>
		/// <param name="liveTime">The live time of the hook (defaults to forever). Use if a hook should only be executed x times.</param>
		public TimeStep(TimeScale timeScale, int interval, int liveTime = LiveTimeForever)
		{
			if (timeScale == null)
			{
				throw new ArgumentNullException(nameof(timeScale));
			}

			if (interval <= 0)
			{
				throw new ArgumentNullException($"Time step interval must be >= 1, but was {interval}.");
			}

			if (liveTime == 0)
			{
				throw new ArgumentException("Live time cannot be zero (must be > 0 for discrete value, < 0, but was, well, zero.");
			}

			TimeScale = timeScale;
			Interval = interval;
			LiveTime = liveTime;
		}

		/// <summary>
		/// An easily readable version of the <see cref="TimeStep"/> constructor.
		/// </summary>
		/// <param name="interval">The time step interval in <see cref="TimeScale"/> units.</param>
		/// <param name="timeScale">The time scale unit.</param>
		/// <param name="liveTime">The live time of the hook (defaults to forever). Use if a hook should only be executed x times.</param>
		/// <returns>A time step with the given properties.</returns>
		public static TimeStep Every(int interval, TimeScale timeScale, int liveTime = LiveTimeForever)
		{
			return new TimeStep(timeScale, interval, liveTime);
		}

		/// <summary>
		/// Deep copy this object.
		/// </summary>
		/// <returns>A deep copy of this object.</returns>
		public object DeepCopy()
		{
			TimeStep copy = new TimeStep(TimeScale, Interval, LiveTime);

			copy.LocalInterval = LocalInterval;
			copy.LocalLiveTime = LocalLiveTime;

			return copy;
		}

		/// <summary>
		/// Check if this time step equals another time steps time step (ignoring local time interval values).
		/// </summary>
		/// <param name="other">The other time step.</param>
		/// <returns>A boolean indicating if this time step equals another time steps time step.</returns>
		public bool StepEquals(ITimeStep other)
		{
			return other != null && TimeScale == other.TimeScale && Interval == other.Interval && LiveTime == other.LiveTime;
		}

		/// <summary>Compares the current instance with another object of the same type and returns an integer that indicates whether the current instance precedes, follows, or occurs in the same position in the sort order as the other object.</summary>
		/// <returns>A value that indicates the relative order of the objects being compared. The return value has these meanings: Value Meaning Less than zero This instance precedes <paramref name="obj" /> in the sort order. Zero This instance occurs in the same position in the sort order as <paramref name="obj" />. Greater than zero This instance follows <paramref name="obj" /> in the sort order. </returns>
		/// <param name="obj">An object to compare with this instance. </param>
		/// <exception cref="T:System.ArgumentException">
		/// <paramref name="obj" /> is not the same type as this instance. </exception>
		public int CompareTo(object obj)
		{
			ITimeStep otherTimeStep = obj as ITimeStep;

			if (otherTimeStep == null)
			{
				throw new ArgumentException($"Object to compare with must be of type {typeof(ITimeStep)} but was {obj.GetType()}.");
			}

			if (otherTimeStep.TimeScale == TimeScale)
			{
				return otherTimeStep.Interval - Interval;
			}

			if (otherTimeStep.TimeScale == TimeScale.Indeterminate || TimeScale == TimeScale.Indeterminate)
			{
				throw new InvalidOperationException($"Cannot compare indeterminate with determinate time scale, attempted to compare this {TimeScale} with other {otherTimeStep.TimeScale}.");
			}

			return TimeScale == TimeScale.Epoch && otherTimeStep.TimeScale == TimeScale.Iteration ? 1 : -1;
		}
	}

	/// <summary>
	/// A time scale, the basic unit for <see cref="ITimeStep"/> (e.g. epoch or iteration).
	/// </summary>
	public class TimeScale
	{
		/// <summary>
		/// The name of this time scale.
		/// </summary>
		public string Name { get; }

		/// <summary>
		/// A time scale for one training epoch (single forward + backward pass on the entire training data, consisting of multiple iterations).
		/// </summary>
		public static readonly TimeScale Epoch = new TimeScale(nameof(Epoch));

		/// <summary>
		/// A time scale for one training iteration (single forward + backward pass on iterator yield).
		/// </summary>
		public static readonly TimeScale Iteration = new TimeScale(nameof(Iteration));

		/// <summary>
		/// A time scale for a start training event.
		/// </summary>
		public static readonly TimeScale Start = new TimeScale(nameof(Start));

		/// <summary>
		/// A time scale for a stop training event.
		/// </summary>
		public static readonly TimeScale Stop = new TimeScale(nameof(Start));

		/// <summary>
		/// A time scale for a resume training event.
		/// </summary>
		public static readonly TimeScale Resume = new TimeScale(nameof(Start));

		/// <summary>
		/// A time scale for a pause training event.
		/// </summary>
		public static readonly TimeScale Pause = new TimeScale(nameof(Start));

		/// <summary>
		/// A time scale that is managed by the callee (e.g. when only invoking once, like for <see cref="ICommand"/>).
		/// </summary>
		public static readonly TimeScale Indeterminate = new TimeScale(nameof(Indeterminate));

		public TimeScale(string name)
		{
			Name = name;
		}

		/// <summary>Returns a string that represents the current object.</summary>
		/// <returns>A string that represents the current object.</returns>
		public override string ToString()
		{
			return Name;
		}
	}
}
