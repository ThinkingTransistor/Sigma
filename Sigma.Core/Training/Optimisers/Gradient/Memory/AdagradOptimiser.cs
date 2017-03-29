/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Optimisers.Gradient.Memory
{
	/// <summary>
	/// An adagrad optimiser which adapts the learning rate for each parameter basted on the sum of past squared gradients.
	/// </summary>
	[Serializable]
	public class AdagradOptimiser : BaseMemoryGradientOptimiser<INDArray>
	{
		/// <summary>
		/// Create an adagrad optimiser which adapts the learning rate for each parameter basted on the sum of past squared gradients.
		/// </summary>
		/// <param name="baseLearningRate">The base learning rate.</param>
		/// <param name="smoothing">The smoothing parameter to the learning adapting </param>
		/// <param name="externalCostAlias">Optionally, the external cost alias to use.</param>
		public AdagradOptimiser(double baseLearningRate, double smoothing = 1E-6, string externalCostAlias = "external_cost") : base("memory_squared_gradient", externalCostAlias)
		{
			Registry.Set("base_learning_rate", baseLearningRate, typeof(double));
			Registry.Set("smoothing", smoothing, typeof(double));
		}

		/// <summary>
		/// Optimise a certain parameter given a certain gradient using a certain computation handler.
		/// </summary>
		/// <param name="paramIdentifier">The parameter identifier (e.g. "3-elementwise.weights").</param>
		/// <param name="parameter">The parameter to optimise.</param>
		/// <param name="gradient">The gradient of the parameter respective to the total cost.</param>
		/// <param name="handler">The handler to use.</param>
		/// <returns>The optimised parameter.</returns>
		protected override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
		{
			double learningRate = Registry.Get<double>("base_learning_rate"), smoothing = Registry.Get<double>("smoothing");
			INDArray squaredGradientSum = GetMemory(paramIdentifier, () => handler.NDArray((long[]) parameter.Shape.Clone()));

			squaredGradientSum = handler.Add(squaredGradientSum, handler.Multiply(gradient, gradient));
			SetMemory(paramIdentifier, squaredGradientSum);

			INDArray adaptedLearningRate = handler.Divide(learningRate, handler.SquareRoot(handler.Add(squaredGradientSum, smoothing)));

			return handler.Add(parameter, handler.Multiply(gradient, handler.Multiply(adaptedLearningRate, -1.0)));
		}

		/// <summary>
		/// Deep copy this object.
		/// </summary>
		/// <returns>A deep copy of this object.</returns>
		public override object DeepCopy()
		{
			return new AdagradOptimiser(Registry.Get<double>("base_learning_rate"), Registry.Get<double>("smoothing"), ExternalCostAlias);
		}
	}
}
