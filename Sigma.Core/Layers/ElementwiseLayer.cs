/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	public class ElementwiseLayer : BaseLayer
	{
		public ElementwiseLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			int size = parameters.Get<int>("size");

			parameters["weights"] = handler.NDArray(size);
			parameters["biases"] = handler.NDArray(size);

			TrainableParameters = new[] { "weights", "biases" };
		}

		public static LayerConstruct New(string name, int size)
		{
			LayerConstruct layer = new LayerConstruct(name, typeof(ElementwiseLayer));

			layer.Parameters["size"] = size;

			return layer;
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass = true)
		{
			
		}
	}
}
