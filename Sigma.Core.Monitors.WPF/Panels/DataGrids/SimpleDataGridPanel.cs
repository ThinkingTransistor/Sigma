using System.Collections.Generic;

namespace Sigma.Core.Monitors.WPF.Panels.DataGrids
{
	public class SimpleDataGridPanel<T> : SigmaDataGridPanel
	{
		public SimpleDataGridPanel(string title) : base(title)
		{
			Items = new List<T>();

			Content.ItemsSource = Items;
		}

		public List<T> Items { get; }
	}
}