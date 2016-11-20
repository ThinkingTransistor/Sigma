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

// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.Control.Tabs
{
	public class TabControlUI<TWindow, TTabWrapper> : UIWrapper<Layout> where TWindow : SigmaWindow where TTabWrapper : UIWrapper<TabItem>
	{
		public TabablzControl InitialTabablzControl { get; set; }

		private readonly TabablzControl _tabControl;

		private Dictionary<string, TTabWrapper> Tabs { get; }

		public TabControlUI (WPFMonitor monitor, App app, string title)
		{
			Tabs = new Dictionary<string, TTabWrapper>();

			if (_tabControl == null)
			{
				_tabControl = new TabablzControl();

				if (InitialTabablzControl == null)
				{
					InitialTabablzControl = _tabControl;
				}
			}

			//TODO: Change with style
			//Restore tabs if they are closed
			_tabControl.ConsolidateOrphanedItems = true;

			//Allow to create new dragged out windows
			_tabControl.InterTabController = new InterTabController() { InterTabClient = new CustomInterTabClient(monitor, app, title) };

			_tabControl.FontFamily = UiResources.FontFamily;
			_tabControl.FontSize = UiResources.P1;

			Content.Content = _tabControl;
		}

		public void AddTab (string header, TTabWrapper tabUI)
		{
			Tabs.Add(header, tabUI);
			_tabControl.Items.Add((TabItem) tabUI);
		}

		private class CustomInterTabClient : IInterTabClient
		{
			private readonly WPFMonitor _monitor;
			private readonly App _app;
			private readonly string _title;

			public CustomInterTabClient (WPFMonitor monitor, App app, string title)
			{
				_monitor = monitor;
				_app = app;
				_title = title;
			}

			public INewTabHost<Window> GetNewHost (IInterTabClient interTabClient, object partition, TabablzControl source)
			{
				TWindow window = Construct(new[] { typeof(WPFMonitor), typeof(App), typeof(string), typeof(bool) }, new object[] { _monitor, _app, _title, false });
				return new NewTabHost<WPFWindow>(window, window.TabControl.InitialTabablzControl);
			}

			public TabEmptiedResponse TabEmptiedHandler (TabablzControl tabControl, Window window)
			{
				window.Close();
				return TabEmptiedResponse.CloseWindowOrLayoutBranch;
			}

			private static TWindow Construct (Type[] paramTypes, object[] paramValues)
			{
				Type t = typeof(TWindow);

				ConstructorInfo ci = t.GetConstructor(
					BindingFlags.Instance | BindingFlags.NonPublic,
					null, paramTypes, null);

				return (TWindow) ci.Invoke(paramValues);
			}
		}

		public TTabWrapper this[string tabname] => Tabs[tabname];
	}
}
