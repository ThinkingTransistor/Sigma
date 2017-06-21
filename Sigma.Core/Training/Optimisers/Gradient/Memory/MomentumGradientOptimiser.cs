/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Optimisers.Gradient.Memory;

namespace Sigma.Core.Training.Optimisers.Gradient
{
	/// <summary>
	/// A momentum gradient optimiser using a basic momentum gradient optimisation algorithm with a certain learning rate and momentum.
	/// The parameter update rule is: 
	///		velocity = velocity * momentum - gradient * baseLearningRate
	///     parameter = parameter + velocity
	/// </summary>
	[Serializable]
	public class MomentumGradientOptimiser : BaseArrayMemoryGradientOptimiser
	{
		/// <summary>
		/// Create a momentum gradient optimiser using a basic momentum gradient optimisation algorithm with a certain learning rate and momentum.
		/// </summary>
		/// <param name="learningRate">The learning rate.</param>
		/// <param name="momentum">The momentum.</param>
		/// <param name="externalCostAlias">The external cost alias.</param>
		public MomentumGradientOptimiser(double learningRate, double momentum, string externalCostAlias = "external_cost") : base("memory_momentum", externalCostAlias) 
		{
			Registry.Set("learning_rate", learningRate, typeof(double));
			Registry.Set("momentum", momentum, typeof(double));
		}

		/// <inheritdoc />
		internal override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
		{
			double learningRate = Registry.Get<double>("learning_rate"), momentum = Registry.Get<double>("momentum");
			INDArray velocity = GetMemory(paramIdentifier, gradient);

			velocity = handler.Add(handler.Multiply(velocity, momentum), handler.Multiply(gradient, learningRate));

			SetProtectedMemory(paramIdentifier, velocity, handler);

			INDArray update = handler.Multiply(velocity, -1.0);

			ExposeParameterUpdate(paramIdentifier, update);

			return handler.Add(parameter, update);
		}

		/// <inheritdoc />
		public override object DeepCopy()
		{
			return new MomentumGradientOptimiser(learningRate: Registry.Get<double>("learning_rate"), momentum: Registry.Get<double>("momentum"));
		}
	}
}
