using System.Windows;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class TabControlFactory : IUIFactory<TabControlUI<SigmaWindow, TabUI>>
	{
		public TabControlFactory(WPFMonitor monitor)
		{
			WpfMonitor = monitor;
		}

		public WPFMonitor WpfMonitor { get; }

		TabControlUI<SigmaWindow, TabUI> IUIFactory<TabControlUI<SigmaWindow, TabUI>>.CreateElement(Application app, Window window,
			params object[] parameters)
		{
			return new TabControlUI<SigmaWindow, TabUI>(WpfMonitor, app, window.Title);
		}
	}
}