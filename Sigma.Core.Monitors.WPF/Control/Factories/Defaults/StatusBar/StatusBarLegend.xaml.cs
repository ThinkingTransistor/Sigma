using System.Windows;
using System.Windows.Media;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Defaults.StatusBar
{
	/// <summary>
	/// Interaction logic for StatusBarLegend.xaml
	/// </summary> 
	public partial class StatusBarLegend
	{
		#region DependencyProperties

		public static readonly DependencyProperty SizeProperty = DependencyProperty.Register(nameof(SizeProperty),
			typeof(double), typeof(StatusBarLegend), new UIPropertyMetadata(18.0));

		public static readonly DependencyProperty LegendColourProperty =
			DependencyProperty.Register(nameof(LegendColourProperty), typeof(Brush), typeof(StatusBarLegend),
				new UIPropertyMetadata(Brushes.White));

		public static readonly DependencyProperty LabelColourProperty =
			DependencyProperty.Register(nameof(LabelColourProperty), typeof(Brush), typeof(StatusBarLegend),
				new UIPropertyMetadata(Brushes.Black));

		public static readonly DependencyProperty TextProperty = DependencyProperty.Register(nameof(TextProperty),
			typeof(string), typeof(StatusBarLegend), new UIPropertyMetadata(""));

		#endregion DependencyProperties

		#region Properties

		public double Size
		{
			get { return (double) GetValue(SizeProperty); }
			set { SetValue(SizeProperty, value); }
		}

		public Brush LegendColour
		{
			get { return (Brush) GetValue(LegendColourProperty); }
			set { SetValue(LegendColourProperty, value); }
		}

		public Brush LabelColour
		{
			get { return (Brush) GetValue(LabelColourProperty); }
			set { SetValue(LabelColourProperty, value); }
		}

		public string Text
		{
			get { return (string) GetValue(TextProperty); }
			set { SetValue(TextProperty, value); }
		}

		#endregion Properties

		public StatusBarLegend()
		{
			InitializeComponent();
		}
	}
}
