/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Optimisers
{
	/// <summary>
	/// A gradient descent optimiser, using the gradient descent algorithm with a certain learning rate on each parameter. 
	/// </summary>
	public class GradientDescentOptimiser : BaseGradientOptimiser
	{
		private readonly double _learningRate;

		/// <summary>
		/// Create a gradient descent optimiser with a certain learning rate.
		/// </summary>
		/// <param name="learningRate">The learning rate.</param>
		/// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
		protected GradientDescentOptimiser(double learningRate, string externalCostAlias = "external_cost") : base(externalCostAlias)
		{
			_learningRate = learningRate;
		}

		protected override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler, IRegistry registry)
		{
			return handler.Add(parameter, handler.Multiply(gradient, -_learningRate));
		}
	}
}
