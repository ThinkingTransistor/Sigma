/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Optimisers.Gradient
{
	/// <summary>
	/// A gradient descent optimiser, using the classic gradient descent algorithm with a certain learning rate on each parameter.
	/// The parameter update rule is: 
	///     parameter = parameter - gradient * learning_rate
	/// </summary>
	[Serializable]
	public class GradientDescentOptimiser : BaseGradientOptimiser
	{
		/// <summary>
		/// Create a gradient descent optimiser with a certain learning rate.
		/// </summary>
		/// <param name="learningRate">The learning rate.</param>
		/// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
		public GradientDescentOptimiser(double learningRate, string externalCostAlias = "external_cost") : base(externalCostAlias)
		{
			Registry.Set("learning_rate", learningRate, typeof(double));
		}

		internal override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
		{
			INDArray update = handler.Multiply(gradient, -Registry.Get<double>("learning_rate"));

			ExposeParameterUpdate(paramIdentifier, update);

			return handler.Add(parameter, update);
		}

		/// <summary>
		/// Deep copy this object.
		/// </summary>
		/// <returns>A deep copy of this object.</returns>
		protected override BaseGradientOptimiser ShallowCopyParameters()
		{
			return new GradientDescentOptimiser(learningRate: Registry.Get<double>("learning_rate"), externalCostAlias: ExternalCostAlias);
		}
	}
}
