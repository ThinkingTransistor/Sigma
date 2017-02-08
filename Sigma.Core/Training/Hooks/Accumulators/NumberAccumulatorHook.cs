/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Accumulators
{
	public class NumberAccumulatorHook : BaseHook
	{
		public NumberAccumulatorHook(string registryEntry, TimeStep timeStep) : base(timeStep, registryEntry)
		{
			
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver"></param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			throw new NotImplementedException();
		}
	}
}
