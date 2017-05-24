using System;
using System.Windows.Media;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Optimisations
{
	/// <summary>
	/// A panel that can easily be used to display a targetmaximisation hook.
	/// </summary>
	public abstract class BitmapHookPanel : BitmapPanel
	{
		///  <summary>
		///      Create a BitmapPanel that can easily be updated by a hook.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed, the title will be used.</param>
		/// <param name="inputWidth">The width of the bitmappanel (not the actual width but the width of the data grid).</param>
		/// <param name="inputHeight">The height of the bitmappanel (not the actual height but the height of the data grid).</param>
		protected BitmapHookPanel(string title, int inputWidth, int inputHeight, object headerContent = null)
			: base(title, inputWidth, inputHeight, headerContent)
		{
			RenderOptions.SetBitmapScalingMode(Content, BitmapScalingMode.NearestNeighbor);
		}

		///  <summary>
		///      Create a BitmapPanel that can easily be updated by a hook. This hook will be automatically attached to a given trainer
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="trainer">The trainer the hook will be applied to.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed, the title will be used.</param>
		/// <param name="inputWidth">The width of the bitmappanel (not the actual width but the width of the data grid).</param>
		/// <param name="inputHeight">The height of the bitmappanel (not the actual height but the height of the data grid).</param>
		/// <param name="hook">A hook. </param>
		protected BitmapHookPanel(string title, int inputWidth, int inputHeight, IHook hook, ITrainer trainer, object headerContent = null)
			: this(title, inputWidth, inputHeight, headerContent)
		{
			if (hook == null) throw new ArgumentNullException(nameof(hook));
			if (trainer == null) throw new ArgumentNullException(nameof(trainer));

			trainer.AddHook(hook);
		}

		/// <summary>
		/// Provide new data to this panel. 
		/// </summary>
		/// <param name="parameterRegistry">The parameter registry of the hook.</param>
		/// <param name="handler">The handler associated with the hooks trainer.</param>
		/// <param name="inputs">The inputs from the hook.</param>
		/// <param name="desiredTargets">The desired targets from the hook.</param>
		public void Update(IRegistry parameterRegistry, IComputationHandler handler, INDArray inputs, INDArray desiredTargets)
		{
			if (Initialised)
			{
				OnReported(parameterRegistry, handler, inputs, desiredTargets);
			}
			else
			{
				throw new InvalidOperationException("Panel not yet initialised.");
			}
		}

		/// <summary>
		/// This method will be called if the panel receives new values from a hook (or an external source in general).
		/// </summary>
		/// <param name="parameterRegistry">The parameter registry of the hook.</param>
		/// <param name="handler">The handler associated with the hooks trainer.</param>
		/// <param name="inputs">The inputs from the hook.</param>
		/// <param name="desiredTargets">The desired targets from the hook.</param>
		protected abstract void OnReported(IRegistry parameterRegistry, IComputationHandler handler, INDArray inputs, INDArray desiredTargets);

		/// <summary>
		/// A target maximisation hook that is responsible for reporting to a <see ref="BitmapHookPanel"/>.
		/// </summary>
		public class VisualTargetMaximisationReporter : TargetMaximisationReporter
		{
			private const string TargetPanel = "mnist_panel";

			/// <summary>
			/// Create a reporter hook that reports the maximisation back to a given panel.
			/// </summary>
			/// <param name="panel">The panel this hook reports back to.</param>
			/// <param name="desiredTargets">The desired target of the maximisation.</param>
			/// <param name="timestep">The timestep this hook executes on.</param>
			public VisualTargetMaximisationReporter(BitmapHookPanel panel, INDArray desiredTargets, ITimeStep timestep) : this(panel, desiredTargets, 0.05, timestep)
			{ }

			/// <summary>
			/// Create a reporter hook that reports the maximisation back to a given panel - the panel has to be specified later on.
			/// </summary>
			/// <param name="desiredTargets">The desired target of the maximisation.</param>
			/// <param name="timestep">The timestep this hook executes on.</param>
			public VisualTargetMaximisationReporter(INDArray desiredTargets, ITimeStep timestep) : this(desiredTargets, 0.05, timestep)
			{ }

			/// <summary>
			/// Create a reporter hook that reports the maximisation back to a given panel - the panel has to be specified later on.
			/// </summary>
			/// <param name="desiredTargets">The desired target of the maximisation.</param>
			/// <param name="desiredCost">The desired cost.</param>
			/// <param name="timestep">The timestep this hook executes on.</param>
			public VisualTargetMaximisationReporter(INDArray desiredTargets, double desiredCost, ITimeStep timestep) : base(desiredTargets, desiredCost, timestep)
			{ }

			/// <summary>
			/// Create a reporter hook that reports the maximisation back to a given panel.
			/// </summary>
			/// <param name="panel">The panel this hook reports back to.</param>
			/// <param name="desiredTargets">The desired target of the maximisation.</param>
			/// <param name="desiredCost">The desired cost.</param>
			/// <param name="timestep">The timestep this hook executes on.</param>
			public VisualTargetMaximisationReporter(BitmapHookPanel panel, INDArray desiredTargets, double desiredCost, ITimeStep timestep) : this(desiredTargets, desiredCost, timestep)
			{
				SetPanel(panel);
			}

			/// <summary>
			/// Set the panel this hook reports back.
			/// </summary>
			/// <param name="panel">The new target panel.</param>
			public void SetPanel(BitmapHookPanel panel)
			{
				ParameterRegistry[TargetPanel] = panel;
			}

			/// <summary>
			/// Handle a successful maximisation.
			/// </summary>
			/// <param name="handler">The computation handler.</param>
			/// <param name="inputs">The inputs.</param>
			/// <param name="desiredTargets">The desired targets.</param>
			protected override void OnTargetMaximisationSuccess(IComputationHandler handler, INDArray inputs, INDArray desiredTargets)
			{
				BitmapHookPanel panel = (BitmapHookPanel)ParameterRegistry[TargetPanel];
				panel.Update(ParameterRegistry, handler, inputs, desiredTargets);
			}
		}
	}
}