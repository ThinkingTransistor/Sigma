/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Modifiers
{
	/// <summary>
	/// A scale value modifier to scale certain values such that x = x * scale.
	/// </summary>
	public class ScaleValueModifier : BaseValueModifier
	{
		/// <summary>
		/// Create a scale value modifier with a certain scale.
		/// </summary>
		/// <param name="scale">The scale.</param>
		public ScaleValueModifier(double scale)
		{
			Registry.Set("scale", scale, typeof(double));
		}

		/// <summary>
		/// Modify a certain ndarray with a certain identifier.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier.</param>
		/// <param name="parameter">The parameter to modify.</param>
		/// <param name="handler">The computation handler to use.</param>
		/// <returns>The modified parameter.</returns>
		public override INDArray Modify(string paramIdentifier, INDArray parameter, IComputationHandler handler)
		{
			return handler.Multiply(parameter, Registry.Get<double>("scale"));
		}
	}
}
