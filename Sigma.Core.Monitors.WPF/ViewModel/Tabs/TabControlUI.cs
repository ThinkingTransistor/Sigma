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

namespace Sigma.Core.Monitors.WPF.ViewModel.Tabs
{
	/// <summary>
	///     This is a tab controller for a specific window and a specific Tab.
	/// </summary>
	/// <typeparam name="TWindow">The window that will be used. A <see cref="SigmaWindow" /> is required. </typeparam>
	/// <typeparam name="TTabWrapper">
	///     An arbitrary tab wrapper can be used as long as it uses <see cref="TabItem" /> from
	///     Dragablz.
	/// </typeparam>
	public class TabControlUI<TWindow, TTabWrapper> : UIWrapper<Layout> where TWindow : SigmaWindow
		where TTabWrapper : UIWrapper<TabItem>
	{
		/// <summary>
		///     The current <see cref="TabablzControl" />.
		/// </summary>
		private readonly TabablzControl _tabControl;

		/// <summary>
		///     Create a new <see cref="TabControlUI{TWindow,TTabWrapper}" />.
		/// </summary>
		/// <param name="monitor">The correct monitor.</param>
		/// <param name="app">The root <see cref="Application" />.</param>
		/// <param name="title">The title of the old (and new) window. </param>
		public TabControlUI(WPFMonitor monitor, Application app, string title)
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

			//TODO: Change with style?
			//Restore tabs if they are closed
			_tabControl.ConsolidateOrphanedItems = true;

			//Allow to create new dragged out windows
			_tabControl.InterTabController = new InterTabController
			{
				InterTabClient = new CustomInterTabClient(monitor, app, title)
			};

			_tabControl.FontFamily = UIResources.FontFamily;
			_tabControl.FontSize = UIResources.P1;

			Content.Content = _tabControl;
		}

		/// <summary>
		///     The initial <see cref="TabablzControl" /> control in order to maintain a
		///     consistent root control.
		/// </summary>
		public TabablzControl InitialTabablzControl { get; set; }

		/// <summary>
		///     Mapping between the tabs as strings and the <see cref="UIWrapper{Tabitem}" />.
		/// </summary>
		private Dictionary<string, TTabWrapper> Tabs { get; }

		/// <summary>
		///     Get a tab by its unique identifier.
		/// </summary>
		/// <param name="tabname">The unique identifier (<see cref="AddTab" />). </param>
		/// <returns>The tab mapped to the given name. </returns>
		public TTabWrapper this[string tabname] => Tabs[tabname];

		/// <summary>
		///     Add a given tab to the UIMapping.
		/// </summary>
		/// <param name="header">
		///     A unique header for the tab. (others get the tab
		///     via this identifier).
		/// </param>
		/// <param name="tabUI">The actual <see cref="TabItem" />.</param>
		public void AddTab(string header, TTabWrapper tabUI)
		{
			Tabs.Add(header, tabUI);
			_tabControl.Items.Add((TabItem) tabUI);
		}

		/// <summary>
		///     This class is the a custom <see cref="IInterTabClient" />. It is
		///     necessary in order to create new <see cref="Window" />s with the
		///     correct constructor (this is done via reflection).
		///     This <see cref="IInterTabClient" /> only works with classes
		///     inheriting from <see cref="SigmaWindow" />.
		/// </summary>
		private class CustomInterTabClient : IInterTabClient
		{
			/// <summary>
			///     The referenced <see cref="App" />.
			/// </summary>
			private readonly Application _app;

			/// <summary>
			///     The referenced <see cref="WPFMonitor" />.
			/// </summary>
			private readonly WPFMonitor _monitor;

			/// <summary>
			///     The title of the current and newly created window.
			/// </summary>
			private readonly string _title;

			/// <summary>
			///     Create a new <see cref="CustomInterTabClient" /> with given attributes.
			///     These attributes are required to create a new <see cref="SigmaWindow" />.
			/// </summary>
			/// <param name="monitor"></param>
			/// <param name="app"></param>
			/// <param name="title"></param>
			public CustomInterTabClient(WPFMonitor monitor, Application app, string title)
			{
				_monitor = monitor;
				_app = app;
				_title = title;
			}

			/// <summary>
			///     Create a new <see cref="SigmaWindow" /> when a tab is dragged out.
			///     Provide a new host window so a tab can be teared from an existing window into
			///     a new window.
			/// </summary>
			/// <param name="interTabClient"></param>
			/// <param name="partition">Provides the partition where the drag operation was initiated.</param>
			/// <param name="source">The source control where a dragging operation was initiated.</param>
			/// <returns></returns>
			public INewTabHost<Window> GetNewHost(IInterTabClient interTabClient, object partition, TabablzControl source)
			{
				TWindow window = Construct(new[] {typeof(WPFMonitor), typeof(Application), typeof(string), typeof(TWindow)},
					new object[] {_monitor, _app, _title, _monitor.Window});
				return new NewTabHost<WPFWindow>(window, window.TabControl.InitialTabablzControl);
			}

			/// <summary>
			///     Called when a tab has been emptied, and thus typically a window needs closing.
			/// </summary>
			/// <param name="tabControl"></param>
			/// <param name="window"></param>
			/// <returns></returns>
			public TabEmptiedResponse TabEmptiedHandler(TabablzControl tabControl, Window window)
			{
				window.Close();
				return TabEmptiedResponse.CloseWindowOrLayoutBranch;
			}

			//TODO: own static helper class?
			/// <summary>
			///     A helper function to call a constructor via reflection
			///     with arbitrary parameters. (Also searches for <see cref="BindingFlags.NonPublic" />
			///     constructors).
			/// </summary>
			/// <param name="paramTypes">An array of the types of the constructor</param>
			/// <param name="paramValues">The objects to pass to the constructor. </param>
			/// <returns></returns>
			private static TWindow Construct(Type[] paramTypes, object[] paramValues)
			{
				Type t = typeof(TWindow);

				ConstructorInfo ci = t.GetConstructor(
					BindingFlags.Instance | BindingFlags.NonPublic,
					null, paramTypes, null);

				return (TWindow) ci.Invoke(paramValues);
			}
		}
	}
}