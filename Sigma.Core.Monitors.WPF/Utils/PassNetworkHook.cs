using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Panels;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public class PassNetworkHook : BaseHook
	{
		private const string DataIdentifier = "data";
		private const string PanelIdentifier = "panel";
		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		public PassNetworkHook(IOutputPanel panel, IDictionary<string, INDArray> block) : base(Core.Utils.TimeStep.Every(1, TimeScale.Iteration), "network.self")
		{
			ParameterRegistry[DataIdentifier] = block;
			ParameterRegistry[PanelIdentifier] = panel;
			InvokeInBackground = true;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			IDictionary<string, INDArray> block = (IDictionary<string, INDArray>)ParameterRegistry[DataIdentifier];
			IOutputPanel panel = (IOutputPanel)ParameterRegistry[PanelIdentifier];

			INetwork network = resolver.ResolveGetSingle<INetwork>("network.self");

			IDataProvider provider = new DefaultDataProvider();
			provider.SetExternalOutputLink("external_default", (targetsRegistry, layer, targetBlock) =>
			{
				panel.SetOutput((INDArray)targetsRegistry["activations"]);
			});

			DataProviderUtils.ProvideExternalInputData(provider, network, block);
			network.Run(Operator.Handler, false);
			DataProviderUtils.ProvideExternalOutputData(provider, network, block);
		}
	}
}