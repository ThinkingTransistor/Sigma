/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers.Cost
{
	/// <summary>
	/// A cost layer that calculates the squared difference between predictions and targets.
	/// </summary>
	[Serializable]
	public class SquaredDifferenceCostLayer : BaseCostLayer
	{
		public SquaredDifferenceCostLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
		}

		protected override INumber CalculateCost(INDArray predictions, INDArray targets, IRegistry parameters, IComputationHandler handler)
		{
			INDArray difference = handler.Subtract(predictions, targets);
			INumber cost = handler.Divide(handler.Sum(handler.Pow(difference, 2)), predictions.Shape[0]);
            // something wrong with predictions * predictions / pow predictions 2 -> wrong result / gradient? maybe with more than one item? see locals debug

			return cost;
		}

		public static LayerConstruct Construct(string name = "#-squaredcost", double importance = 1.0, string externalTargetsAlias = "external_targets", string externalCostAlias = "external_cost")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(SquaredDifferenceCostLayer));

			return InitialiseBaseConstruct(construct, importance, externalTargetsAlias, externalCostAlias);
		}
	}
}
