/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Training.Hooks.Accumulators
{
	/// <summary>
	/// An accumulator hook for accumulating any number over a certain time step (this hook's time step).
	/// </summary>
	[Serializable]
	public class NumberAccumulatorHook : BaseHook
	{
		public NumberAccumulatorHook(string registryEntry, TimeStep timeStep, int resetEvery = -1, int resetInterval = 0) : this(registryEntry, registryEntry.Replace('.', '_') + "_accumulated", timeStep, resetEvery, resetInterval)
		{
		}

		public NumberAccumulatorHook(string registryEntry, string resultEntry, TimeStep timeStep, int resetEvery = -1, int resetInterval = 0) : base(timeStep, registryEntry)
		{
			ParameterRegistry["registry_entry"] = registryEntry;
			ParameterRegistry["shared_result_entry"] = resultEntry;
			ParameterRegistry["reset_interval"] = resetInterval;
			ParameterRegistry["reset_every"] = resetEvery;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			string registryEntry = ParameterRegistry.Get<string>("registry_entry");
			string resultEntry = ParameterRegistry.Get<string>("shared_result_entry");

			double value = resolver.ResolveGetSingle<double>(registryEntry);
			double accumulatedValue = resolver.ResolveGetSingleWithDefault<double>(resultEntry, 0.0);

			int currentInterval = HookUtils.GetCurrentInterval(registry, TimeStep.TimeScale);
			int resetInterval = ParameterRegistry.Get<int>("reset_interval");
			int resetEvery = ParameterRegistry.Get<int>("reset_every");

			if (currentInterval == resetInterval || resetEvery > 0 && currentInterval % resetEvery == 0)
			{
				accumulatedValue = 0.0;
			}

			resolver.ResolveSet(resultEntry, value + accumulatedValue, addIdentifierIfNotExists: true);
		}
	}
}
