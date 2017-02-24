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
	/// A clip value modifier that clips all parameters to a certain range.
	/// </summary>
	public class ClipValueModifier : BaseValueModifier
	{
		/// <summary>
		/// Create a clip value modifier that clips all values to a certain range (defaults to [-1, 1]).
		/// </summary>
		/// <param name="minValue">The optional minimum value.</param>
		/// <param name="maxValue">The optional maximum value.</param>
		public ClipValueModifier(double minValue = -1.0, double maxValue = 1.0)
		{
			Registry["min_value"] = minValue;
			Registry["max_value"] = maxValue;
		}

		public override INDArray Modify(string paramIdentifier, INDArray parameter, IComputationHandler handler)
		{
			INumber minValue = handler.Number(Registry.Get<double>("min_value"));
			INumber maxValue = handler.Number(Registry.Get<double>("max_value"));

			return handler.Clip(parameter, minValue, maxValue);
		}
	}
}
