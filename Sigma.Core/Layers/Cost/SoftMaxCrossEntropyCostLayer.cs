/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers.Cost
{
	public class SoftMaxCrossEntropyCostLayer : CostLayer
	{
		/// <summary>
		/// Create a base layer with a certain unique name.
		/// </summary>
		/// <param name="name">The unique name of this layer.</param>
		/// <param name="parameters">The parameters to this layer.</param>
		/// <param name="handler">The handler to use for ndarray parameter creation.</param>
		public SoftMaxCrossEntropyCostLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
		}

		protected override INumber CalculateCost(INDArray predictions, INDArray targets, IRegistry parameters, IComputationHandler handler)
		{
			predictions = handler.SoftMax(predictions);

			// cross entropy
			// |	a	|  |	   b		 |
			// y * ln(x) + (1 - y) * ln(1 - x)

			INDArray a = handler.Multiply(predictions, handler.Log(targets));
			INDArray b = handler.Multiply(handler.Subtract(1, targets), handler.Log(handler.Subtract(1, predictions)));
			INDArray cost = handler.Add(a, b);

			return handler.Divide(handler.Sum(cost), cost.Length);
		}

		public static LayerConstruct Construct(string name = "#-softmaxce")
		{
			return new LayerConstruct(name, typeof(SoftMaxCrossEntropyCostLayer));
		}
	}
}

