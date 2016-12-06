using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Defaults
{
	public class RootPanelFactory : IUIFactory<DockPanel>
	{
		public DockPanel CreatElement(App app, Window window, params object[] parameters)
		{
			return new DockPanel { LastChildFill = true };
		}
	}
}
