using System;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.Parameterisation
{
	/// <summary>
	/// A hook that updates a given IParameterVisualiser on a given TimeStep.
	/// </summary>
	[Serializable]
	public class PollParameterHook : BaseHook
	{
		/// <summary>
		/// The identifier for the currently active visualiser.
		/// </summary>
		protected const string VisualiserIdentifier = "visualiser";

		/// <summary>
		/// Create a hook with a certain time step
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="visualiser">The visualisers that will be updated.</param>
		public PollParameterHook(ITimeStep timestep, IParameterVisualiser visualiser) : base(timestep)
		{
			ParameterRegistry[VisualiserIdentifier] = visualiser;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			((IParameterVisualiser) ParameterRegistry[VisualiserIdentifier]).Read();
		}
	}
}