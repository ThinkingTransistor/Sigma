using System.Windows;
using System.Windows.Media;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls;

namespace Sigma.Core.Monitors.WPF.Model.UI.StatusBar
{
	/// <summary>
	///     TODO:
	/// </summary>
	// This is a class (not a struct) because it may be inherited from. 
	public class StatusBarLegendInfo
	{
		public Color LegendColor { get; set; }

		public string Text { get; set; }

		public StatusBarLegend Apply(StatusBarLegend statusBarLegend)
		{
			//yes, a new SolidColorBrush has to be created every time
			//otherwise thread problems may arise. 
			statusBarLegend.LegendColour = new SolidColorBrush(LegendColor);

			statusBarLegend.Text = Text;

			statusBarLegend.Margin = new Thickness(0, 0, 5, 0);

			return statusBarLegend;
		}
	}
}