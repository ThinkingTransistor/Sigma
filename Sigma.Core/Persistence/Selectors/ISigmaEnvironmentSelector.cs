/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Training;

namespace Sigma.Core.Persistence.Selectors
{
	public interface ISigmaEnvironmentSelector : ISelector<SigmaEnvironment>
	{
	}

	/// <summary>
	/// An individual trainer component for <see cref="ITrainer"/> data selection (keep / discard).
	/// </summary>
	[Flags]
	public enum SigmaEnvironmentComponent
	{
		/// <summary>
		/// Nothing (except the environment name, which is the minimum state and included by default).
		/// </summary>
		None = 0,

		/// <summary>
		/// All attached monitors.
		/// </summary>
		Monitors = 1 << 0,

		/// <summary>
		/// All attached trainers.
		/// </summary>
		Trainers = 1 << 1,

		/// <summary>
		/// All attached trainers, monitors and additional runtime data (pending requests, hook queues, execution states).
		/// </summary>
		RuntimeState = Monitors | Trainers,

		/// <summary>
		/// Everything.
		/// </summary>
		All = int.MaxValue
	}
}
