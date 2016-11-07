using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;

namespace Sigma.Core.Monitors.WPF.View.TitleBar
{
	public class TitleBarItem : Button
	{
		//#region DependencyProperties

		//public static readonly DependencyProperty TextProperty = DependencyProperty.Register("Text", typeof(string), typeof(TitleBarItem), new UIPropertyMetadata("null"));

		//#endregion DependencyProperties

		//#region Properties

		///// <summary>
		///// The text that is displayed for the <see cref="TitleBarItem"/>.
		///// </summary>
		//public string Text
		//{
		//	get { return (string) GetValue(TextProperty); }
		//	set { SetValue(TextProperty, value); }
		//}

		//#endregion Properties

		public TitleBarItem(string text) : base()
		{
			//Background = Brushes.Transparent;
			//BorderBrush = Brushes.Transparent;
			//FontSize = 15;
			Content = text;

			
			
			//Style = Application.Current.FindResource("WindowCommandsPopupBoxStyle") as Style;
			//StackPanel panel = new StackPanel();
			//panel.Children.Add(new Button() { Content = text });
			//base.Content = panel;
		}
	}
}
