/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

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
	public class InputLayer : BaseLayer
	{
		public InputLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			// external to indicate that these parameters are not only external (which should already be indicate with the InputsExternal flag in the layer construct and buffer)
			//	but also that they mark the boundaries of the entire network (thereby external to the network, not only external as in external source)
			ExpectedInputs = new[] { parameters.Get<string>("external_input_alias") };
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			buffer.Outputs["default"]["activations"] = buffer.Inputs[buffer.Parameters.Get<string>("external_input_alias")]["activations"];
		}

		public static LayerConstruct Construct(params long[] shape)
		{
			return Construct("#-inputs", shape);
		}

		public static LayerConstruct Construct(string name, params long[] shape)
		{
			return Construct(name, "external_default", shape);
		}

		public static LayerConstruct Construct(string name, string inputAlias, params long[] shape)
		{
			NDArrayUtils.CheckShape(shape);
			LayerConstruct construct = new LayerConstruct(name, typeof(InputLayer));

			construct.ExternalInputs = new[] { inputAlias };
			construct.Parameters["external_input_alias"] = inputAlias;
			construct.Parameters["shape"] = shape;
			construct.Parameters["size"] = ArrayUtils.Product(shape);

			return construct;
		}
	}
}
