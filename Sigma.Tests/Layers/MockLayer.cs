/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.Utils;

namespace Sigma.Tests.Layers
{
	public class MockLayer : BaseLayer
	{
		public MockLayer(string name, IRegistry parameters, IComputationHandler handler) : base(name, parameters, handler)
		{
		}

		public override void Run(ILayerBuffer buffer, IComputationHandler handler, bool trainingPass)
		{
		}

		public static LayerConstruct Construct()
		{
			return new LayerConstruct("#-mocktestlayer", typeof(MockLayer));
		}
	}
}
