using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Monitors.WPF.Panels.Games.TicTacToe
{
	public class TicTacToePanel : GenericPanel<TicTacToeField>, IOutputPanel, IInputPanel, IPassNetworkReceiver, IDisposable
	{
		protected readonly ITrainer Trainer;
		protected PassNetworkHook PassNetworkHook;
		protected Action InvokePass;
		protected IDictionary<string, INDArray> Block;

		/// <summary>
		/// a list of moveorders that contains the order of possible moves.
		/// </summary>
		protected List<int> _moveOrder;


		//TODO: document
		//only works with sigmawindow
		public TicTacToePanel(string title, ITrainer trainer, object headerContent = null) : base(title, headerContent)
		{
			Trainer = trainer;
			Handler = trainer.Operator.Handler;
			Input = this;
		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected override void OnInitialise(SigmaWindow window)
		{
			Content = new TicTacToeField(Handler, Monitor);

			_moveOrder = new List<int>(9);

			Block = new Dictionary<string, INDArray>();
			PassNetworkHook = new PassNetworkHook(this, Block, TimeStep.Every(1, TimeScale.Epoch));
			PassNetworkHook.On(new ExternalCriteria(registerHoldFunc: action => InvokePass = action));

			UpdateBlock();

			Content.AiMove += AIMoveRequest;

			Trainer.AddGlobalHook(PassNetworkHook);

			//Content.FieldChange += FieldChange;
			IsReady = true;
		}

		private void AIMoveRequest(object sender, EventArgs e)
		{
			if (UpdateBlock())
			{
				InvokePass();
			}
		}

		private bool UpdateBlock()
		{
			INDArray bruteForcedValues = GeneratePossibleMoves(Values);
			if (bruteForcedValues == null)
			{
				return false;
			}

			Block["inputs"] = bruteForcedValues;
			Block["targets"] = Handler.NDArray(bruteForcedValues.Shape[0], 1, 3);

			return true;
		}

		/// <Inheritdoc />
		public IInputPanel Input { get; set; }
		/// <Inheritdoc />
		public void SetInputReference(INDArray values)
		{
			throw new InvalidOperationException();
		}

		/// <Inheritdoc />
		public void SetOutput(INDArray output)
		{
			throw new InvalidOperationException();
		}

		/// <Inheritdoc />
		public IComputationHandler Handler { get; set; }

		/// <Inheritdoc />
		public bool IsReady { get; private set; }

		/// <Inheritdoc />
		public INDArray Values
		{
			get { return Content.Field; }
		}

		protected virtual INDArray GeneratePossibleMoves(INDArray values)
		{
			List<int[]> possibleMoves = new List<int[]>();
			int[] valuesInts = values.GetDataAs<int>().Data;
			for (int i = 0; i < values.Length; i++)
			{
				if (valuesInts[i] == 0)
				{
					int[] valuesIntsClone = (int[])valuesInts.Clone();
					valuesIntsClone[i] = 1;
					possibleMoves.Add(valuesIntsClone);

					_moveOrder.Add(i);
				}
			}

			if (possibleMoves.Count > 0)
			{
				int[] moves = possibleMoves.Aggregate(ArrayUtils.Concatenate);
				return Handler.NDArray(moves, possibleMoves.Count, 1, 9);
			}

			return null;
		}

		/// <summary>
		/// This method accepts a network pass and processes.
		/// </summary>
		/// <param name="array">The array that is the response from the pass.</param>
		public void ReceivePass(INDArray array)
		{
			array = Handler.FlattenTimeAndFeatures(array);
			array = Handler.RowWise(array, Handler.SoftMax);

			float[][] elements = Handler.RowWiseTransform(array, subarray => subarray.GetDataAs<float>().Data);

			int maxIndex = 0;
			float maxScore = float.NegativeInfinity;

			for (int i = 0; i < elements.Length; i++)
			{
				float[] currentMove = elements[i];
				float score = currentMove[1] + currentMove[2] * 2;

				if (score > maxScore)
				{
					maxIndex = i;
				}
			}

			if (maxIndex < _moveOrder.Count)
			{
				int maxPosInGrid = _moveOrder[maxIndex];
				_moveOrder.Clear();
				Content.Dispatcher.Invoke(() => Content.SetIndex(maxPosInGrid / 3, maxPosInGrid % 3, Content.AiTicTacToePlayer));
			}
		}

		// TODO: detachpanel
		//on window close remove from events / call dispose automatically when closing the window

		public void Dispose()
		{
			Content.AiMove -= AIMoveRequest;
			//Content.FieldChange -= FieldChange;
		}


	}
}