using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Panels.DataGrids
{
	public abstract class SigmaDataGridPanel : SigmaPanel
	{
		protected SigmaDataGridPanel(string title) : base(title)
		{
			Content = new DataGrid
			{
				IsReadOnly = true
			};

			base.Content = Content;
		}

		public new DataGrid Content { get; }
	}
}