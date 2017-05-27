using System;
using System.Linq;
using System.Windows;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Panels.Optimisations;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults;
using Sigma.Core.Training;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Utils.Defaults.MNIST
{
	/// <summary>
	/// A target maximisation panel optimised for MNIST.
	/// </summary>
	public class MnistBitmapHookPanel : BitmapHookPanel
	{
		///  <summary>
		///		Create a BitmapPanel that can easily be updated by a hook. This hook will be automatically attached to a given trainer.
		///		This panel is optimised for MNIST with one hot of 10 and a given number.
		///  </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="timestep">The timestep this panel updates.</param>
		/// <param name="trainer">The trainer the hook will be applied to.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed, the title will be used.</param>
		/// <param name="number">The number this panel tries to visualise.</param>
		public MnistBitmapHookPanel(string title, int number, ITrainer trainer, ITimeStep timestep, object headerContent = null) : base(title, 28, 28, headerContent)
		{
			if (trainer == null) throw new ArgumentNullException(nameof(trainer));

			UseLoadingIndicator = true;
			VisualTargetMaximisationReporter hook = new VisualTargetMaximisationReporter(this, trainer.Operator.Handler.NDArray(ArrayUtils.OneHot(number, 10), 10L), timestep);
			trainer.AddGlobalHook(hook);
		}

		/// <inheritdoc />
		protected override void OnReported(IRegistry parameterRegistry, IComputationHandler handler, INDArray inputs, INDArray desiredTargets)
		{
			HideLoadingIndicator();

			float[] targetData = inputs.GetDataAs<float>().Data;
			float min = targetData.Min(), max = targetData.Max();
			float range = max - min;

			Content.Dispatcher.Invoke(() => RenderRectangle<float>(inputs, r => 0, g => 0, b => (byte)(Math.Pow((b - min) / range, 1.5f) * 255), a => 0xff, 0, 0));
		}
	}
}