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

			// TODO replace with proper cross entropy cost
			INumber cost = handler.Divide(handler.Sum(handler.Multiply(targets, handler.Log(predictions))), -predictions.Length);

			return cost;
		}

		public static LayerConstruct Construct(string name = "#-softmaxce", double importance = 1.0, string externalTargetsAlias = "external_targets", string externalCostAlias = "external_cost")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(SoftMaxCrossEntropyCostLayer));

			return InitialiseConstruct(construct, importance, externalTargetsAlias, externalCostAlias);
		}
	}
}

