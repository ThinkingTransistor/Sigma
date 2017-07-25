/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Optimisers.Gradient.Memory
{
	/// <summary>
	/// A base memory gradient optimiser with array protection for implementing gradient optimisers that require some form of array memory (e.g. over past gradients, param updates).
	/// </summary>
	public abstract class BaseArrayMemoryGradientOptimiser : BaseMemoryGradientOptimiser<INDArray>
	{
		/// <summary>
		/// Create a base memory gradient optimiser with an optional external output cost alias to use. 
		/// </summary>
		/// <param name="memoryIdentifier">The memory identifier (useful for parametrisation, so be somewhat descriptive).</param>
		/// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
		protected BaseArrayMemoryGradientOptimiser(string memoryIdentifier, string externalCostAlias = "external_cost") : base(memoryIdentifier, externalCostAlias)
		{

		}

		/// <summary>
		/// Set the array memory entry for a certain parameter identifier and automatically move it to session "limbo" for better (and safer) caching.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="value">The memory entry to set.</param>
		/// <returns>The given memory entry (for convenience).</returns>
		protected INDArray SetProtectedMemory(string paramIdentifier, INDArray value, IComputationHandler handler)
		{
			if (!IsInMemory(paramIdentifier))
			{
				Registry.Get<Dictionary<string, INDArray>>(MemoryIdentifier).Add(paramIdentifier, value);
			}
			else
			{
				var memory = Registry.Get<Dictionary<string, INDArray>>(MemoryIdentifier);

				handler.FreeLimbo(memory[paramIdentifier]); // free previous value from session limbo

				memory[paramIdentifier] = value;
			}

			handler.MarkLimbo(value);

			return value;
		}
	}
}
