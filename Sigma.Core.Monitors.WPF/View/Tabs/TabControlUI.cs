using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;
using Dragablz;
using Dragablz.Dockablz;

namespace Sigma.Core.Monitors.WPF.View.Tabs
{
	internal class TabControlUI : UIWrapper<Layout>
	{
		private TabablzControl tabControl;

		private List<UIWrapper<TabItem>> tabs;

		public TabControlUI() : base()
		{
			tabs = new List<UIWrapper<TabItem>>();
			tabControl = new TabablzControl();

			content.Content = tabControl;
		}

		public void AddTab(UIWrapper<TabItem> tabUI)
		{
			tabs.Add(tabUI);
			tabControl.Items.Add((TabItem) tabUI);
		}
	}
}
