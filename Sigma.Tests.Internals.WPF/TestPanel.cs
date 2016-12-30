using System.Collections.Generic;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.Panels;

namespace Sigma.Tests.Internals.WPF
{
	public class TestPanel : SigmaPanel
	{
		public TestPanel(string name) : base(name)
		{
			List<object> column1 = new List<object> {"test1", "test2"};

			DataGrid grid = new DataGrid();

			grid.Columns.Add(new DataGridTextColumn {Header = "column1"});
			grid.ItemsSource = column1;
			Content = grid;
		}
	}
}