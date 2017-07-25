using System;
using Sigma.Core.Architecture;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Processors
{
	public class TimeSeriesGeneratorHook : BaseHook
	{
		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="requiredRegistryEntries">The required global registry entries.</param>
		public TimeSeriesGeneratorHook(ITimeStep timestep) : base(timestep, "network.self")
		{
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			INetwork network = resolver.ResolveGetSingle<INetwork>("network.self");

			//DataUtils.ProvideExternalInputData(network, DataUtils.MakeBlock());
		}
	}
}
