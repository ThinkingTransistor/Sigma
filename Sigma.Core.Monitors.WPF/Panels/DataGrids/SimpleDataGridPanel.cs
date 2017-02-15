/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.ObjectModel;

namespace Sigma.Core.Monitors.WPF.Panels.DataGrids
{
	public class SimpleDataGridPanel<T> : SigmaDataGridPanel
	{
		public ObservableCollection<T> Items { get; }

		public SimpleDataGridPanel(string title, object content = null) : base(title, content)
		{
			Items = new ObservableCollection<T>();

			Content.ItemsSource = Items;
		}

	}
}