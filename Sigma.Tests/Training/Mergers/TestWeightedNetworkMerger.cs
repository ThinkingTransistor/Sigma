using NUnit.Framework;
using Sigma.Core.Architecture;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Utils;

namespace Sigma.Tests.Training.Mergers
{
	public class TestWeightedNetworkMerger
	{
		[TestCase]
		public void TestWeightedNetworkMergerMerge()
		{
			INetworkMerger merger = new WeightedNetworkMerger(3, 8);
			INetwork netA = NetworkMergerTestUtils.GenerateNetwork(1);
			INetwork netB = NetworkMergerTestUtils.GenerateNetwork(9);

			merger.AddMergeEntry("layers.*.weights");

			merger.Merge(netA, netB);

			IRegistryResolver resolverA = new RegistryResolver(netA.Registry);
			IRegistryResolver resolverB = new RegistryResolver(netB.Registry);

			INDArray weightsA = resolverA.ResolveGet<INDArray>("layers.*.weights")[0];
			INDArray weightsB = resolverB.ResolveGet<INDArray>("layers.*.weights")[0];

			float firstValueA = weightsA.GetValue<float>(0, 0);
			float firstValueB = weightsB.GetValue<float>(0, 0);

			// the first value will change
			Assert.AreEqual(6.82, System.Math.Round(firstValueA * 100) / 100);

			// the second net may not be changed
			Assert.AreEqual(9, firstValueB);

			merger.RemoveMergeEntry("layers.*.weights");

			merger.Merge(netA, netB);
			weightsA = resolverA.ResolveGet<INDArray>("layers.*.weights")[0];
			firstValueA = weightsA.GetValue<float>(0, 0);

			// may not change
			Assert.AreEqual(6.82, System.Math.Round(firstValueA * 100) / 100);
		}
	}
}