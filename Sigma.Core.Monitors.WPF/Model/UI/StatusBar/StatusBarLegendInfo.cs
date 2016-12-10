using System.Windows;
using System.Windows.Media;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
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
		public Color? ForegroundColor { get; set; }

		/// <summary>
		/// The displayed text and value for the mapping.  
		/// </summary>
		public string Name { get; }

		public StatusBarLegendInfo(string name)
		{
			Name = name;
		}

		public StatusBarLegendInfo(string name, MaterialColour colour, PrimaryColour primary = PrimaryColour.Primary700, AccentColour accent = AccentColour.Accent400) : this(name)
		{
			LegendColor = MaterialDesignValues.GetColour(colour, primary);
			//TODO: fix me
			ForegroundColor = MaterialDesignValues.GetColour(colour, accent);
		}

		public StatusBarLegend Apply(StatusBarLegend statusBarLegend)
		{
			//yes, a new SolidColorBrush has to be created every time
			//otherwise thread problems may arise. 
			statusBarLegend.LegendColour = new SolidColorBrush(LegendColor);

			statusBarLegend.Text = Name;

			statusBarLegend.Margin = new Thickness(0, 0, 5, 0);

			return statusBarLegend;
		}
	}
}