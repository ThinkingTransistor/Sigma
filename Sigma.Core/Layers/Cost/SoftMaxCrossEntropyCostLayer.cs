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
	public class SoftMaxCrossEntropyCostLayer : BaseCostLayer
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
			predictions = handler.RowWise(predictions, handler.SoftMax);

			INDArray logPredictions = handler.Log(predictions);
			INDArray a = handler.Multiply(targets, logPredictions);

			INDArray inverseTargets = handler.Subtract(1, targets);
			INDArray inversePredictions = handler.Subtract(1, predictions);
			INDArray b = handler.Multiply(inverseTargets, handler.Log(inversePredictions));

			INumber cost = handler.Divide(handler.Sum(handler.Add(a, b)), -predictions.Shape[0]);

			return cost;
		}

		public static LayerConstruct Construct(string name = "#-softmaxcecost", double importance = 1.0, string externalTargetsAlias = "external_targets", string externalCostAlias = "external_cost")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(SoftMaxCrossEntropyCostLayer));

			return InitialiseBaseConstruct(construct, importance, externalTargetsAlias, externalCostAlias);
		}
	}
}