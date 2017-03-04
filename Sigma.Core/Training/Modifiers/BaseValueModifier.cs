/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Modifiers
{
	/// <summary>
	/// A base value modifier which transforms all numbers to ndarrays for convenience if they are to be treated identically.
	/// Note: If you want to handle numbers and ndarrays separately, create your own <see cref="IValueModifier"/> implementation.
	/// </summary>
	[Serializable]
	public abstract class BaseValueModifier : IValueModifier
	{
		/// <summary>
		/// The registry containing this value modifier's parameter for external inspection and modification (if applicable).
		/// </summary>
		public IRegistry Registry { get; } = new Registry(tags: "value_modifier");

		/// <summary>
		/// Modify a certain number with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		public INumber Modify(string paramIdentifier, INumber parameter, IComputationHandler handler)
		{
			return handler.AsNumber(Modify(paramIdentifier, handler.AsNDArray(parameter), handler));
		}

		/// <summary>
		/// Modify a certain number with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		public double Modify(string paramIdentifier, double parameter, IComputationHandler handler)
		{
			return Modify(paramIdentifier, handler.NDArray(new[] {parameter}, 1L, 1L), handler).GetValue<double>(0, 0);
		}

		/// <summary>
		/// Modify a certain ndarray with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		public abstract INDArray Modify(string paramIdentifier, INDArray parameter, IComputationHandler handler);
	}
}
