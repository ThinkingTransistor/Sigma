using System.Windows;
using System.Windows.Controls.Primitives;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Sigma
{
	public class StatusBarFactory : IUIFactory<UIElement>
	{
		private double _height;

		public StatusBarFactory(double height)
		{
			_height = height;
		}

		public UIElement CreatElement(App app, Window window)
		{
			StatusBar statusBar = new StatusBar
			{
				Height = _height,
				Background = UIResources.AccentColorBrush
			};


			return statusBar;
		}
	}
}