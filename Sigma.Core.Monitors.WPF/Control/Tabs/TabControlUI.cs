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
	public class TabControlUi<T> : UiWrapper<Layout> where T : SigmaWindow
	{
		public TabablzControl InitialTabablzControl { get; set; }

		private TabablzControl _tabControl;

		private List<UiWrapper<TabItem>> _tabs;

		public List<UiWrapper<TabItem>> Tabs
		{
			get
			{
				return _tabs;
			}
		}

		public TabControlUi (WPFMonitor monitor, App app, string title) : base()
		{
			_tabs = new List<UiWrapper<TabItem>>();

			if (_tabControl == null)
			{
				_tabControl = new TabablzControl();

				if (InitialTabablzControl == null)
				{
					InitialTabablzControl = _tabControl;
				}
			}

			//Restore tabs if they are closed
			_tabControl.ConsolidateOrphanedItems = true;

			//Allow to create new dragged out windows
			_tabControl.InterTabController = new InterTabController() { InterTabClient = new CustomInterTabClient(monitor, app, title) };

			_tabControl.FontFamily = UiResources.FontFamily;
			_tabControl.FontSize = UiResources.P1;

			Content.Content = _tabControl;
		}

		public void AddTab (UiWrapper<TabItem> tabUi)
		{
			_tabs.Add(tabUi);
			_tabControl.Items.Add((TabItem) tabUi);
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
				T window = Construct(new Type[] { typeof(WPFMonitor), typeof(App), typeof(string), typeof(bool) }, new object[] { _monitor, _app, _title, false });
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
