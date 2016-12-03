using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Panels.DataGrids
{
	public abstract class SigmaDataGridPanel : SigmaPanel
	{
		public new DataGrid Content { get; }

		protected SigmaDataGridPanel(string title) : base(title)
		{
			Content = new DataGrid
			{
				IsReadOnly = true,
			};

			base.Content = Content;
		}
	}
}