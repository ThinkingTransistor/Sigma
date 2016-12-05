using System.Windows;
using Sigma.Core.Monitors.WPF.Control.TitleBar;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Sigma
{
	public class TitleBarFactory : IUIFactory<TitleBarControl>
	{
		public Thickness Margin { get; }
		public Thickness Padding { get; }

		public TitleBarFactory() : this(new Thickness(0), new Thickness(0)) { }

		public TitleBarFactory(Thickness margin, Thickness padding)
		{
			Margin = margin;
			Padding = padding;
		}

		public TitleBarControl CreatElement(App app, Window window)
		{
			TitleBarControl titleBarControl = new TitleBarControl
			{
				Margin = Margin,
				Padding = Padding
			};


			return titleBarControl;
		}
	}
}