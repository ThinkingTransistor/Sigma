/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers.Feedforward
{
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
			INDArray inputActivations = buffer.Inputs["default"].Get<INDArray>("activations");
		}

		public static LayerConstruct Construct(int size, string name = "#-elementwise")
		{
			LayerConstruct construct = new LayerConstruct(name, typeof(ElementwiseLayer));

			construct.Parameters["size"] = size;

			return construct;
		}
	}
}
