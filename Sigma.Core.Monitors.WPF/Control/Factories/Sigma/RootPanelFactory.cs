using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Control.Factories.Sigma
{
	public class RootPanelFactory : IUIFactory<DockPanel>
	{
		public DockPanel CreatElement(App app, Window window)
		{
			return new DockPanel { LastChildFill = true };
		}
	}
}
