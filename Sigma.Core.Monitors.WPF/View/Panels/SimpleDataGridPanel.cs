using System.Collections.Generic;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public class SimpleDataGridPanel<T> : SigmaPanel
	{
		public List<T> Items { get; }

		public new DataGrid Content { get; }

		public SimpleDataGridPanel(string title) : base(title)
		{
			Items = new List<T>();

			Content = new DataGrid
			{
				ItemsSource = Items,
				IsReadOnly = true
			};

			base.Content = Content;
		}
	}
}