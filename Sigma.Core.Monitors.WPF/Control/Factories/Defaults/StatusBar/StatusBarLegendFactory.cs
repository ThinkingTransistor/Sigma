using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Defaults.StatusBar
{
	//public struct StatusBarLegend
	//{
	//	/// <summary>
	//	/// The colour of the square. 
	//	/// </summary>
	//	public Color Colour { get; set; }
	//	/// <summary>
	//	/// The size of the square.
	//	/// </summary>
	//	public double Size { get; set; }
	//	/// <summary>
	//	/// The next next to the legend square. 
	//	/// </summary>
	//	public string Text { get; set; }
	//}

	public class StatusBarLegendFactory : IUIFactory<UIElement>
	{
		public UIElement CreatElement(App app, Window window, params object[] parameters)
		{
			Grid grid = new Grid();

			//grid.RowDefinitions.Add(new RowDefinition());
			//grid.ColumnDefinitions.Add(new ColumnDefinition() {Width = });
			//grid.ColumnDefinitions.Add(new ColumnDefinition());

			return grid;
		}
	}
}
