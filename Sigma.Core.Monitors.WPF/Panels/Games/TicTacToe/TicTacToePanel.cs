using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Panels.Games.TicTacToe
{
	public class TicTacToePanel : GenericPanel<TicTacToeField>, IOutputPanel, IInputPanel
	{
		private readonly IComputationHandler _handler;

		//TODO: document
		//only works with sigmawindow
		public TicTacToePanel(string title, IComputationHandler handler, object headerContent = null) : base(title, headerContent)
		{
			_handler = handler;
			Input = this;
		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected override void OnInitialise(SigmaWindow window)
		{
			Content = new TicTacToeField(_handler, Monitor);
			IsReady = true;
		}

		/// <Inheritdoc />
		public IInputPanel Input { get; set; }
		/// <Inheritdoc />
		public void SetInputReference(INDArray values)
		{
			throw new System.NotImplementedException();
		}

		/// <Inheritdoc />
		public void SetOutput(INDArray output)
		{
			throw new System.NotImplementedException();
		}

		/// <Inheritdoc />
		public IComputationHandler Handler { get; set; }

		public bool IsReady { get; private set; }

		/// <Inheritdoc />
		public INDArray Values
		{
			get { return Content.Field; }
		}
	}
}