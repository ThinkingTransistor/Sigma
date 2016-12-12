using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class RootPanelFactory : IUIFactory<DockPanel>
	{
		public DockPanel CreatElement(Application app, Window window, params object[] parameters)
		{
			return new DockPanel { LastChildFill = true };
		}
	}
}