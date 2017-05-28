using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.Panels.DataGrids;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Providers;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	//TODO: HACK: only for presentation
	public class Guess
	{
		public int Digit { get; set; }
		public string Probability { get; set; }
	}

	//TODO: move to own class
	public class NumberPanel : SimpleDataGridPanel<Guess>, IOutputPanel
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
			Content.Padding = new Thickness(20, 0, 20, 0);
			Content.FontSize = 25;
			Handler = trainer.Operator.Handler;

			Trainer = trainer;
			_guesses = new List<Guess>(10);
		}

		private List<Guess> _guesses;

		public void SetInputReference(INDArray values)
		{
			Items.Clear();
			for (int i = 0; i < 10; i++)
			{
				Guess guess = new Guess();
				_guesses.Add(guess);
				Items.Add(guess);
			}
			Values = values;
			//TODO: check if hook already added, remove if...
			IDictionary<string, INDArray> block = new Dictionary<string, INDArray>();
			block.Add("inputs", Values);
			block.Add("targets", Handler.NDArray(1, 1, 10));
			Trainer.AddGlobalHook(new PassNetworkHook(this, block));
		}

		public void SetOutput(INDArray output)
		{
			output = Handler.SoftMax(Handler.Multiply(output, 10)); // TODO remove hack and fix for very small numbers
			KeyValuePair<double, int>[] sorted = output.GetDataAs<double>().Data.Select((x, i) => new KeyValuePair<double, int>(x, i)).OrderByDescending(x => x.Key).ToArray();

			string text = "";

			for (int i = 0; i < sorted.Length; i++)
			{
				double confidence = Math.Round(sorted[i].Key * 10000) / 100;
				int number = sorted[i].Value;
				_guesses[i].Probability = $"{confidence:00.000}";
				_guesses[i].Digit = number;
				//guesses.Add(new Guess { Accuracy = Math.Round(accuracy.Key * 100), Number = accuracy.Value });
			}

			Content.Dispatcher.Invoke(() =>
			{
				Items.Clear();
				Items.AddRange(_guesses);
			});

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
					_outputPanel.SetInputReference(Values);
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

			OutputPanel = outputPanel;

			Content.InputChangedEvent += UpdateValues;
		}

		private static double[] DrawCanvasValuesSingle(RectangleCanvas canvas)
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
			}
		}

		public void Dispose()
		{
			Content.InputChangedEvent -= UpdateValues;
		}

	}
}