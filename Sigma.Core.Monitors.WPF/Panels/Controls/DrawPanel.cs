using System;
using System.Collections.Generic;
using System.Windows.Controls;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	//TODO: move to own class
	public class NumberPanel : GenericPanel<TextBox>, IOutputPanel
	{
		public ITrainer Trainer { get; }

		protected INDArray Values { get; private set; }

		public IComputationHandler Handler { get; set; }

		public IInputPanel Input { get; set; }

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public NumberPanel(string title, ITrainer trainer, object headerContent = null) : base(title, headerContent)
		{
			Handler = trainer.Operator.Handler;

			//TODO: style
			Content = new TextBox();

			Content.FontSize = 72;
			Content.Text = "?";
			Trainer = trainer;
		}

		public void SetInputReference(INDArray values)
		{
			//TODO: check if hook already added, remove if...
			IDictionary<string, INDArray> block = new Dictionary<string, INDArray>();
			block.Add("inputs", Values);
			block.Add("targets", Handler.NDArray(1, 1, 10));
			Trainer.AddGlobalHook(new PassNetworkHook(this, block));
		}

		public void SetOutput(INDArray output)
		{
			output = Handler.SoftMax(output);

			double max = double.NegativeInfinity;
			int maxNum = -1;
			Console.WriteLine(output);
			for (int i = 0; i < 10; i++)
			{
				double current = output.GetValue<double>(i);
				if (current > max)
				{
					max = current;
					maxNum = i;
				}
			}

			Content.Text = maxNum.ToString();
		}

		private class PassNetworkHook : BaseHook
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
				IDictionary<string, INDArray> block = (IDictionary<string, INDArray>) ParameterRegistry[DataIdentifier];
				IOutputPanel panel = (IOutputPanel) ParameterRegistry[PanelIdentifier];


				INetwork network = resolver.ResolveGetSingle<INetwork>("network.self");

				IDataProvider provider = new DefaultDataProvider();
				provider.SetExternalOutputLink("external_default", (targetsRegistry, layer, targetBlock) =>
				{
					panel.SetOutput((INDArray) targetsRegistry["activations"]);
				});

				DataProviderUtils.ProvideExternalInputData(provider, network, block);
				network.Run(Operator.Handler, trainingPass: false);
				DataProviderUtils.ProvideExternalOutputData(provider, network, block);
			}
		}
	}

	public class DrawPanel : GenericPanel<DrawCanvas>, IInputPanel, IDisposable
	{
		private IOutputPanel _outputPanel;
		public IComputationHandler Handler { get; set; }

		public INDArray Values { get; }

		public IOutputPanel OutputPanel
		{
			get { return _outputPanel; }
			set
			{
				_outputPanel = value;
				if (_outputPanel != null)
				{
					_outputPanel.Input = this;
				}
			}
		}

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public DrawPanel(string title, ITrainer trainer, int drawWidth, int drawHeight, int drawSize, IOutputPanel outputPanel = null, object headerContent = null) : base(title, headerContent)
		{
			Handler = trainer.Operator.Handler;
			OutputPanel = outputPanel;

			Content = new DrawCanvas
			{
				GridWidth = drawWidth,
				GridHeight = drawHeight,
				PointSize = drawSize
			};

			Content.UpdateRects();

			double[,] oldVals = Content.GetValues();
			double[] newVals = DrawCanvasValuesSingle(Content);
			Values = Handler.NDArray(newVals, 1, 1, oldVals.GetLength(0), oldVals.GetLength(1));

			Content.InputChangedEvent += UpdateValues;
		}

		private double[] DrawCanvasValuesSingle(DrawCanvas canvas)
		{
			double[,] vals = canvas.GetValues();
			double[] newVals = new double[vals.Length];
			for (int row = 0; row < vals.GetLength(0); row++)
			{
				for (int column = 0; column < vals.GetLength(1); column++)
				{
					newVals[row * vals.GetLength(0) + column] = vals[row, column];
				}
			}
			return newVals;
		}

		protected virtual void UpdateValues(DrawCanvas canvas)
		{
			double[] newVals = DrawCanvasValuesSingle(canvas);
			lock (Values)
			{
				for (int i = 0; i < newVals.Length; i++)
				{
					Values.SetValue(newVals[i], NDArrayUtils.GetIndices(i, Values.Shape, Values.Strides));
				}

				OutputPanel?.UpdateValues(Values);
			}
		}

		public void Dispose()
		{
			Content.InputChangedEvent -= UpdateValues;
		}

	}
}