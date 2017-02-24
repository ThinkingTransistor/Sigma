/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Optimisers
{
	/// <summary>
	/// A momentum gradient optimiser using a basic momentum gradient optimisation algorithm with a certain learning rate and momentum.
	/// The parameter update rule is: 
	///		velocity = velocity * momentum - gradient * learningRate
	///     parameter = parameter + velocity
	/// </summary>
	public class MomentumGradientOptimiser : BaseGradientOptimiser
	{
		public MomentumGradientOptimiser(double learningRate, double momentum)
		{
			Registry.Set("learning_rate", learningRate, typeof(double));
			Registry.Set("momentum", momentum, typeof(double));
			Registry.Set("velocities", new Dictionary<string, INDArray>());
		}
		
		protected override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
		{
			Dictionary<string, INDArray> velocities = Registry.Get<Dictionary<string, INDArray>>("velocities");
			double learningRate = Registry.Get<double>("learning_rate"), momentum = Registry.Get<double>("momentum");
			INDArray velocity;

			if (!velocities.TryGetValue(paramIdentifier, out velocity))
			{
				velocity = parameter;
			}

			velocity = handler.Add(handler.Multiply(velocity, momentum), handler.Multiply(gradient, -learningRate));
			velocities[paramIdentifier] = velocity;

			return handler.Add(velocity, parameter);
		}

		public override object DeepCopy()
		{
			return new MomentumGradientOptimiser(learningRate: Registry.Get<double>("learning_rate"), momentum: Registry.Get<double>("momentum"));
		}
	}
}
