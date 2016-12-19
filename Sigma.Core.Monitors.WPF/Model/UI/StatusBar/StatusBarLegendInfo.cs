using System.Windows;
using System.Windows.Media;
using MaterialDesignColors;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;
using static Sigma.Core.Monitors.WPF.Model.UI.Resources.MaterialDesignValues;

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

		public StatusBarLegendInfo(string name, MaterialColour colour, PrimaryColour primaryColour = PrimaryColour.Primary700) : this(name, GetColour(colour, primaryColour), GetForegroundColor(colour, primaryColour)) { }

		public StatusBarLegendInfo(string name, Swatch swatch, PrimaryColour primaryColour = PrimaryColour.Primary700) : this(name, GetColour(swatch, primaryColour), GetForegroundColor(swatch, primaryColour)) { }

		public StatusBarLegendInfo(string name, MaterialColour colour, AccentColour accentColour) : this(name, GetColour(colour, accentColour), GetForegroundColor(colour, accentColour)) { }

		public StatusBarLegendInfo(string name, Swatch swatch, AccentColour accentColour) : this(name, GetColour(swatch, accentColour), GetForegroundColor(swatch, accentColour)) { }

		public StatusBarLegendInfo(string name, Color legendColor, Color? foregroundColor = null) : this(name)
		{
			LegendColor = legendColor;
			ForegroundColor = foregroundColor;
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