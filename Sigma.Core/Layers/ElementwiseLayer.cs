/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Layers
{
	public class ElementwiseLayer : BaseLayer
	{
		public ElementwiseLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
			int size = parameters.Get<int>("size");

			parameters["weights"] = handler.Create(size);
			parameters["biases"] = handler.Create(size);

			TrainableParameters = new[] { "weights", "biases" };
		}

		public static LayerConstruct New(string name, int size)
		{
			LayerConstruct layer = new LayerConstruct(name, typeof(ElementwiseLayer));

			layer.Parameters["size"] = size;

			return layer;
		}

		public override void Run(AliasRegistry inputs, IRegistry parameters, AliasRegistry outputs, IComputationHandler handler, bool trainingPass = true)
		{

		}
	}
}
