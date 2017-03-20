using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.View.CustomControls.Panels.Control;

namespace Sigma.Core.Monitors.WPF.Panels.Controls
{
	public class DrawPanel : GenericPanel<DrawCanvas>, IInputPanel, IDisposable
	{
		public IComputationHandler handler { get; set; }

		public INDArray Values { get; private set; }

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="SigmaPanel.Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="headerContent">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		public DrawPanel(string title, IComputationHandler handler, int drawWidth, int drawHeight, int drawSize, object headerContent = null) : base(title, headerContent)
		{
			Content = new DrawCanvas
			{
				GridWidth = drawWidth,
				GridHeight = drawHeight,
				PointSize = drawSize
			};

			Content.UpdateRects();

			double[] newVals = DrawCanvasValuesSingle(Content);
			Values = handler.NDArray(newVals);

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

			for (int i = 0; i < newVals.Length; i++)
			{
				Values.SetValue(newVals[i], NDArrayUtils.GetIndices(i, Values.Shape, Values.Strides));
			}
		}

		public void Dispose()
		{
			Content.InputChangedEvent -= UpdateValues;
		}

	}
}