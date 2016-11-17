/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using Dragablz;
using Dragablz.Dockablz;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.View;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.Control.Tabs
{
	public class TabControlUI<T> : UIWrapper<Layout> where T : SigmaWindow
	{
		public TabablzControl InitialTabablzControl { get; set; }

		private TabablzControl tabControl;

		private List<UIWrapper<TabItem>> tabs;

		public List<UIWrapper<TabItem>> Tabs
		{
			get
			{
				return tabs;
			}
		}

		public TabControlUI (WPFMonitor monitor, App app, string title) : base()
		{
			tabs = new List<UIWrapper<TabItem>>();

			if (tabControl == null)
			{
				tabControl = new TabablzControl();

				if (InitialTabablzControl == null)
				{
					InitialTabablzControl = tabControl;
				}
			}

			//Restore tabs if they are closed
			tabControl.ConsolidateOrphanedItems = true;

			//Allow to create new dragged out windows
			tabControl.InterTabController = new InterTabController() { InterTabClient = new CustomInterTabClient(monitor, app, title) };

			tabControl.FontFamily = UIResources.FontFamily;
			tabControl.FontSize = UIResources.P1;

			content.Content = tabControl;
		}

		public void AddTab (UIWrapper<TabItem> tabUI)
		{
			tabs.Add(tabUI);
			tabControl.Items.Add((TabItem) tabUI);
		}

		private class CustomInterTabClient : IInterTabClient
		{
			private WPFMonitor monitor;
			private App app;
			private string title;

			public CustomInterTabClient (WPFMonitor monitor, App app, string title)
			{
				this.monitor = monitor;
				this.app = app;
				this.title = title;
			}

			public INewTabHost<Window> GetNewHost (IInterTabClient interTabClient, object partition, TabablzControl source)
			{
				T window = Construct(new Type[] { typeof(WPFMonitor), typeof(App), typeof(string), typeof(bool) }, new object[] { monitor, app, title, false });
				return new NewTabHost<WPFWindow>(window, window.TabControl.InitialTabablzControl);
			}

			public TabEmptiedResponse TabEmptiedHandler (TabablzControl tabControl, Window window)
			{
				window.Close();
				return TabEmptiedResponse.CloseWindowOrLayoutBranch;
			}

			private static T Construct (Type[] paramTypes, object[] paramValues)
			{
				Type t = typeof(T);

				ConstructorInfo ci = t.GetConstructor(
					BindingFlags.Instance | BindingFlags.NonPublic,
					null, paramTypes, null);

				return (T) ci.Invoke(paramValues);
			}
		}
	}
}
