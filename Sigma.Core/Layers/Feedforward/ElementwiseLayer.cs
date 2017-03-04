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

namespace Sigma.Core.Layers.Feedforward
{
	/// <summary>
	/// An element-wise layer with element-wise weights and connections. 
	/// </summary>
	[Serializable]
	public class ElementwiseLayer : BaseLayer
	{
		public ElementwiseLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			int size = parameters.Get<int>("size");

			parameters["weights"] = handler.NDArray(size);
			parameters["bias"] = handler.Number(0);

			TrainableParameters = new[] { "weights", "bias" };
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			INDArray activations = buffer.Inputs["default"].Get<INDArray>("activations");
			INDArray weights = buffer.Parameters.Get<INDArray>("weights");
			INumber bias = buffer.Parameters.Get<INumber>("bias");

			activations = handler.RowWise(activations, row => handler.Add(handler.Multiply(row, weights), bias));

			buffer.Outputs["default"]["activations"] = activations;
		}

		public static LayerConstruct Construct(int size, string activation = "tanh", string name = "#-elementwise")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(ElementwiseLayer));

			construct.Parameters["size"] = size;
			construct.Parameters["activation"] = activation;

			construct.ValidateEvent += (sender, args) =>
			{
				if (Convert.ToDecimal(args.Self.Inputs["default"].Parameters["size"]) != Convert.ToDecimal(args.Self.Parameters["size"]))
				{
					throw new InvalidNetworkArchitectureException(
						$"Element-wise layer must be connected to input layer of input size, but own (\"{args.Self.Name}\") size {args.Self.Parameters["size"]} " +
						$"does not match size of default input (\"{args.Self.Inputs["default"].Name}\") {args.Self.Inputs["default"].Parameters["size"]}.");
				}
			};

			return construct;
		}
	}
}
