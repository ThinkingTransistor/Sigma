using System.Collections.Generic;

namespace Sigma.Core.Monitors.WPF.View.Panels.DataGrids
{
	public class SimpleDataGridPanel<T> : SigmaDataGridPanel
	{
		public List<T> Items { get; }

		public SimpleDataGridPanel(string title) : base(title)
		{
			Items = new List<T>();

			Content.ItemsSource = Items;
		}
	}
}