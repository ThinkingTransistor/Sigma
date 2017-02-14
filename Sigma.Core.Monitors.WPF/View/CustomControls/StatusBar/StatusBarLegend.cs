/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.View.CustomControls.StatusBar
{
	public class StatusBarLegend : Control
	{
		static StatusBarLegend()
		{
			DefaultStyleKeyProperty.OverrideMetadata(typeof(StatusBarLegend),
				new FrameworkPropertyMetadata(typeof(StatusBarLegend)));
		}

		#region DependencyProperties

		public static readonly DependencyProperty SizeProperty = DependencyProperty.Register(nameof(Size),
			typeof(double), typeof(StatusBarLegend), new UIPropertyMetadata(18.0));

		public static readonly DependencyProperty LegendColourProperty =
			DependencyProperty.Register(nameof(LegendColour), typeof(Brush), typeof(StatusBarLegend),
				new UIPropertyMetadata(Brushes.White));

		public static readonly DependencyProperty LabelColourProperty =
			DependencyProperty.Register(nameof(LabelColour), typeof(Brush), typeof(StatusBarLegend),
				new UIPropertyMetadata(Brushes.Black));

		public static readonly DependencyProperty TextProperty = DependencyProperty.Register(nameof(Text),
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
	}
}