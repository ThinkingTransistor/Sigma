using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Layers;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.Training;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;

namespace Sigma.Tests.Training.Mergers
{
	public class NetworkMergerTestUtils
	{
		private static SigmaEnvironment _environment;
		private static int _count;

		public static INetwork GenerateNetwork(double number)
		{
			if (_environment == null)
			{
				_environment = SigmaEnvironment.Create("TestAverageNetworkMergerEnvironment");
			}

			ITrainer trainer = _environment.CreateTrainer($"trainer{_count++}");

			Network net = new Network();
			net.Architecture = InputLayer.Construct(2, 2) + FullyConnectedLayer.Construct(2 * 2) + OutputLayer.Construct(2);

			trainer.Network = net;
			trainer.AddInitialiser("*.weights", new ConstantValueInitialiser(number));

			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.Initialise(new CpuFloat32Handler());

			SigmaEnvironment.Clear();

			return net;
		}
	}
}