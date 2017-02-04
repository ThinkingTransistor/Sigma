/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporting
{
	/// <summary>
	/// A hook that reports the current progress (e.g. cost) to the console.
	/// </summary>
	public class ProgressReporterHook : BasePassiveHook
	{
		/// <summary>
		/// Create a passive hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		public ProgressReporterHook(ITimeStep timestep, params string[] requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}

		/// <summary>
		/// Create a passive hook with a certain time step and set of required global registry entries.
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The set of required global registry entries.</param>
		public ProgressReporterHook(ITimeStep timestep, ISet<string> requiredRegistryEntries) : base(timestep, requiredRegistryEntries)
		{
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		public override void Invoke(IRegistry registry)
		{
			throw new System.NotImplementedException("Yay, method was called.");
		}
	}
}
