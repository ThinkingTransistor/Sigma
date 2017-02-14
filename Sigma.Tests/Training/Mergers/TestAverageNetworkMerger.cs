using System.Diagnostics;
using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Layers;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Utils;

namespace Sigma.Tests.Training.Mergers
{
	public class TestAverageNetworkMerger
	{
		[TestCase]
		public void TestAverageNetworkMergerMerge()
		{
			INetworkMerger merger = new AverageNetworkMerger();
			INetwork netA = NetworkMergerTestUtils.GenerateNetwork(1);
			INetwork netB = NetworkMergerTestUtils.GenerateNetwork(5);

			merger.AddMergeEntry("layers.*.weights");

			merger.Merge(netA, netB);

			IRegistryResolver resolverA = new RegistryResolver(netA.Registry);
			IRegistryResolver resolverB = new RegistryResolver(netB.Registry);

			INDArray weightsA = resolverA.ResolveGet<INDArray>("layers.*.weights")[0];
			INDArray weightsB = resolverB.ResolveGet<INDArray>("layers.*.weights")[0];
		
			float firstValueA = weightsA.GetValue<float>(0, 0);
			float firstValueB = weightsB.GetValue<float>(0, 0);

			// the first value will change
			Assert.AreEqual(3, firstValueA);

			// the second net may not be changed
			Assert.AreEqual(5, firstValueB);

			merger.RemoveMergeEntry("layers.*.weights");

			merger.Merge(netA, netB);
			weightsA = resolverA.ResolveGet<INDArray>("layers.*.weights")[0];
			firstValueA = weightsA.GetValue<float>(0, 0);

			Assert.AreEqual(3, firstValueA);
		}
	}
}