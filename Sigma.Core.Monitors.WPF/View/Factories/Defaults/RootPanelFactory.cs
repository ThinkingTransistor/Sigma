/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class RootPanelFactory : IUIFactory<DockPanel>
	{
		public DockPanel CreateElement(Application app, Window window, params object[] parameters)
		{
			return new DockPanel {LastChildFill = true};
		}
	}
}