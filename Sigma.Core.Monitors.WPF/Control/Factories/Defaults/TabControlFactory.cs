using System.Windows;
using Sigma.Core.Monitors.WPF.Control.Tabs;
using Sigma.Core.Monitors.WPF.View.Tabs;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Defaults
{
	public class TabControlFactory : IUIFactory<TabControlUI<SigmaWindow, TabUI>>
	{
		public WPFMonitor WpfMonitor { get; }

		public TabControlFactory(WPFMonitor monitor)
		{
			WpfMonitor = monitor;
		}

		TabControlUI<SigmaWindow, TabUI> IUIFactory<TabControlUI<SigmaWindow, TabUI>>.CreatElement(App app, Window window, params object[] parameters)
		{
			return new TabControlUI<SigmaWindow, TabUI>(WpfMonitor, app, window.Title);
		}
	}
}
