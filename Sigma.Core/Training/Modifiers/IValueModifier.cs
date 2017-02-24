/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Modifiers
{
	/// <summary>
	/// A value modifier that modifies ndarrays and numbers according to some specification.
	/// </summary>
	public interface IValueModifier
	{
		/// <summary>
		/// The registry containing this value modifier's parameter for external inspection and modification (if applicable).
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Modify a certain ndarray with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		INDArray Modify(string paramIdentifier, INDArray parameter, IComputationHandler handler);

		/// <summary>
		/// Modify a certain number with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		INumber Modify(string paramIdentifier, INumber parameter, IComputationHandler handler);
	}
}
