using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.ViewModel.CustomControls
{
	/// <summary>
	///     Follow steps 1a or 1b and then 2 to use this custom control in a XAML file.
	///     Step 1a) Using this custom control in a XAML file that exists in the current project.
	///     Add this XmlNamespace attribute to the root element of the markup file where it is
	///     to be used:
	///     xmlns:MyNamespace="clr-namespace:Sigma.Core.Monitors.WPF.Control.CustomControls"
	///     Step 1b) Using this custom control in a XAML file that exists in a different project.
	///     Add this XmlNamespace attribute to the root element of the markup file where it is
	///     to be used:
	///     xmlns:MyNamespace="clr-namespace:Sigma.Core.Monitors.WPF.Control.CustomControls;assembly=Sigma.Core.Monitors.WPF.Control.CustomControls"
	///     You will also need to add a project reference from the project where the XAML file lives
	///     to this project and Rebuild to avoid compilation errors:
	///     Right click on the target project in the Solution Explorer and
	///     "Add Reference"->"Projects"->[Browse to and select this project]
	///     Step 2)
	///     Go ahead and use your control in the XAML file.
	///     <MyNamespace:StatusBarLegend />
	/// </summary>
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