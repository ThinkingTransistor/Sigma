using Sigma.Core.Monitors.WPF.View.Panels;
using System.Collections.Generic;
using System.Windows.Controls;

namespace Sigma.Tests.Internals.WPF
{
	public class TestPanel : SigmaPanel
	{
		public TestPanel(string name) : base(name)
		{
			var column1 = new List<object> { "test1", "test2" };

			var grid = new DataGrid();

			grid.Columns.Add(new DataGridTextColumn { Header = "column1" });
			grid.ItemsSource = column1;
			Content = grid;
		}
	}
}