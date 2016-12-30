/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	/// <summary>
	/// An input layer, i.e. the inputs to this layer are supplied externally.
	/// </summary>
	public class OutputLayer : BaseLayer
	{
		public OutputLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			// external to indicate that these parameters are not only external (which should already be indicate with the InputsExternal flag in the layer construct and buffer)
			//	but also that they mark the boundaries of the entire network (thereby external to the network, not only external as in external source)
			ExpectedOutputs = new[] { "external" }; 
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			buffer.Outputs["external"]["activations"] = buffer.Inputs["default"]["activations"];
		}

		public static LayerConstruct Construct(params long[] shape)
		{
			return Construct("#-outputs", shape);
		}

		public static LayerConstruct Construct(string name, params long[] shape)
		{
			NDArrayUtils.CheckShape(shape);

			LayerConstruct construct = new LayerConstruct(name, typeof(OutputLayer));

			construct.OutputsExternal = true;
			construct.Parameters["shape"] = shape;

			return construct;
		}
	}
}
