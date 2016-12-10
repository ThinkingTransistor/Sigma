using System.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class TitleBarFactory : IUIFactory<TitleBarControl>
	{
		public TitleBarFactory() : this(new Thickness(0), new Thickness(0))
		{
		}

		public TitleBarFactory(Thickness margin, Thickness padding)
		{
			Margin = margin;
			Padding = padding;
		}

		public Thickness Margin { get; }
		public Thickness Padding { get; }

		public TitleBarControl CreatElement(App app, Window window, params object[] parameters)
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