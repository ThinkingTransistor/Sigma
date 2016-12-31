using System.Collections.ObjectModel;

namespace Sigma.Core.Monitors.WPF.Panels.DataGrids
{
	public class SimpleDataGridPanel<T> : SigmaDataGridPanel
	{
		public ObservableCollection<T> Items { get; }

		public SimpleDataGridPanel(string title) : base(title)
		{
			Items = new ObservableCollection<T>();

			Content.ItemsSource = Items;
		}

	}
}