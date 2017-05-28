using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// An interface that can respond to a network pass (<see cref="PassNetworkHook"/>).
	/// </summary>
	public interface IPassNetworkReceiver
	{
		/// <summary>
		/// This method accepts a network pass and processes.
		/// </summary>
		/// <param name="array">The array that is the response from the pass.</param>
		void ReceivePass(INDArray array);
	}

	/// <summary>
	/// A hook that passes as a network and reports the received output to a <see cref="IPassNetworkReceiver"/>.
	/// </summary>
	public class PassNetworkHook : BaseHook
	{
		/// <summary>
		/// An identifier for the parameter registry. Identifier for the reference to the data.
		/// </summary>
		private const string DataIdentifier = "data";
		/// <summary>
		/// An identifier for the parameter registry. Identifier for the reference to the receiver to report.
		/// </summary>
		private const string ReceiverIdentifier = "receiver";

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		public PassNetworkHook(IPassNetworkReceiver receiver, IDictionary<string, INDArray> block, ITimeStep timeStep) : base(timeStep, "network.self")
		{
			ParameterRegistry[DataIdentifier] = block;
			ParameterRegistry[ReceiverIdentifier] = receiver;
			InvokeInBackground = true;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			IDictionary<string, INDArray> block = (IDictionary<string, INDArray>) ParameterRegistry[DataIdentifier];
			IPassNetworkReceiver receiver = (IPassNetworkReceiver) ParameterRegistry[ReceiverIdentifier];

			INetwork network = resolver.ResolveGetSingle<INetwork>("network.self");

			IDataProvider provider = new DefaultDataProvider();
			provider.SetExternalOutputLink("external_default", (targetsRegistry, layer, targetBlock) => { receiver.ReceivePass((INDArray) targetsRegistry["activations"]); });

			DataProviderUtils.ProvideExternalInputData(provider, network, block);
			network.Run(Operator.Handler, false);
			DataProviderUtils.ProvideExternalOutputData(provider, network, block);
		}
	}
}