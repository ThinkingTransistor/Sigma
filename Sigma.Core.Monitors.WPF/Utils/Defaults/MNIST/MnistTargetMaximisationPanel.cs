using System;
using System.Linq;
using System.Windows.Media;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Panels;
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Utils.Defaults.MNIST
{
	public class MnistTargetMaximisationPanel : BitmapPanel
	{
		protected class MnistTargetMaximisationReporter : TargetMaximisationReporter
		{
			private const string TargetPanel = "mnist_panel";
			private const string Number = "number";

			public MnistTargetMaximisationReporter(MnistTargetMaximisationPanel panel, ITrainer trainer, int number, ITimeStep timestep) : this(panel, trainer, number, 0.05, timestep)
			{
			}

			public MnistTargetMaximisationReporter(MnistTargetMaximisationPanel panel, ITrainer trainer, int number, double desiredCost, ITimeStep timestep) : base(trainer.Operator.Handler.NDArray(ArrayUtils.OneHot(number, 10), 10L), desiredCost, timestep)
			{
				ParameterRegistry[TargetPanel] = panel;
				ParameterRegistry[Number] = number;
			}

			/// <summary>
			/// Handle a successful maximisation.
			/// </summary>
			/// <param name="handler">The computation handler.</param>
			/// <param name="inputs">The inputs.</param>
			/// <param name="desiredTargets">The desired targets.</param>
			protected override void OnTargetMaximisationSuccess(IComputationHandler handler, INDArray inputs, INDArray desiredTargets)
			{
				MnistTargetMaximisationPanel panel = (MnistTargetMaximisationPanel)ParameterRegistry[TargetPanel];
				panel.UpdateNumber((int)ParameterRegistry[Number], inputs);
			}
		}

		/// <summary>
		///     Create a BitmapPanel with a given title, width, and height.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed, the title will be used.</param>
		/// <param name="number">The target number.</param>
		public MnistTargetMaximisationPanel(string title, int number, int inputWidth, int inputHeight, ITrainer trainer, ITimeStep timestep, object headerContent = null)
			: base(title, inputWidth, inputHeight, headerContent)
		{
			RenderOptions.SetBitmapScalingMode(Content, BitmapScalingMode.NearestNeighbor);
			trainer.AddGlobalHook(new MnistTargetMaximisationReporter(this, trainer, number, timestep));
		}

		private void UpdateNumber(int number, INDArray target)
		{
			if (Initialised)
			{
				float[] targetData = target.GetDataAs<float>().Data;
				float min = targetData.Min(), max = targetData.Max();
				float range = max - min;

				Content.Dispatcher.Invoke(() => RenderRectangle<float>(target, r => 0, g => 0, b => (byte)(Math.Pow((b - min) / range, 1.5f) * 255), a => 0xff, 0, 0));
			}
		}
	}
}