/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Accumulators
{
	public class NumberAccumulatorHook : BaseHook
	{
		public NumberAccumulatorHook(string registryEntry, TimeStep timeStep) : this(registryEntry, registryEntry.Replace('.', '_') + "_accumulated", timeStep)
		{
		}

		public NumberAccumulatorHook(string registryEntry, string resultEntry, TimeStep timeStep) : base(timeStep, registryEntry)
		{
			ParameterRegistry["registry_entry"] = registryEntry;
			ParameterRegistry["shared_result_entry"] = resultEntry;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			string registryEntry = ParameterRegistry.Get<string>("registry_entry");
			string resultEntry = ParameterRegistry.Get<string>("shared_result_entry");

			double value = resolver.ResolveGetSingle<double>(registryEntry);
			double accumulatedValue = resolver.ResolveGetSingleWithDefault<double>(resultEntry, 0.0);

			resolver.ResolveSet(resultEntry, value + accumulatedValue);
		}
	}
}
