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
	public class InputLayer : BaseLayer
	{
		public InputLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			ExpectedInputs = new[] {"external"};
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
			buffer.Outputs["default"]["activations"] = buffer.Inputs["external"]["activations"];
		}

		public static LayerConstruct Construct(params long[] shape)
		{
			return Construct("inputs#", shape);
		}

		public static LayerConstruct Construct(string name, params long[] shape)
		{
			NDArrayUtils.CheckShape(shape);
			LayerConstruct construct = new LayerConstruct(name, typeof(InputLayer));

			construct.InputsExternal = true;
			construct.Parameters["shape"] = shape;

			return construct;
		}
	}
}
