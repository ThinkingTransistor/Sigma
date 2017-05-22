using System;
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
			{ }

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

		protected MnistTargetMaximisationReporter[] reporters = new MnistTargetMaximisationReporter[10];

		/// <summary>
		///     Create a BitmapPanel with a given title, width, and height.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public MnistTargetMaximisationPanel(string title, ITrainer trainer, ITimeStep timestep, object headerContent = null) : base(title, 28 * 3, 28 * 4, headerContent)
		{
			RenderOptions.SetBitmapScalingMode(Content, BitmapScalingMode.NearestNeighbor);
			for (int i = 0; i < reporters.Length; i++)
			{
				reporters[i] = new MnistTargetMaximisationReporter(this, trainer, i, timestep);
				trainer.AddGlobalHook(reporters[i]);
			}
		}

		public void UpdateNumber(int number, INDArray target)
		{
			if (Initialised)
			{
				int column = number % 3;
				int row = number / 3;

				Content.Dispatcher.Invoke(() => RenderRectangle<float>(target, r => 0, g => 0, b => (byte)(255 - b * 255), a => 0xff, column * 28, row * 28));
			}
		}
	}
}