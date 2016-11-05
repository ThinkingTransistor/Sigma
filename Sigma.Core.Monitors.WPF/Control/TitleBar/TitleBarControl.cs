using MahApps.Metro.Controls;

namespace Sigma.Core.Monitors.WPF.Control.TitleBar
{
	public class TitleBarControl : WindowCommands
	{
		public TitleBarControl()
		{
			ShowLastSeparator = false;
		}

		//#region DependencyProperties

		//public static readonly DependencyProperty ChildrenProperty = DependencyProperty.Register("Text", typeof(string), typeof(TitleBarItem), new UIPropertyMetadata("null"));

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
	}
}
