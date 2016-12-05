/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using Dragablz.Dockablz;
using MahApps.Metro.Controls.Dialogs;
using Sigma.Core.Monitors.WPF.Control.Factories;
using Sigma.Core.Monitors.WPF.Control.Factories.Sigma;
using Sigma.Core.Monitors.WPF.Control.Tabs;
using Sigma.Core.Monitors.WPF.Control.TitleBar;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.View.Tabs;
using Sigma.Core.Monitors.WPF.View.TitleBar;
using CharacterCasing = System.Windows.Controls.CharacterCasing;
using HorizontalAlignment = System.Windows.HorizontalAlignment;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	public class SigmaWindow : WPFWindow
	{
		#region DependencyProperties

		// ReSharper disable once InconsistentNaming
		public static readonly DependencyProperty DefaultGridSizeProperty = DependencyProperty.Register("DefaultGridSize", typeof(GridSize), typeof(WPFWindow), new UIPropertyMetadata(new GridSize(3, 4)));

		#endregion DependencyProperties

		#region Properties

		/// <summary>
		/// The DefaultGridSize for each newly created <see cref="System.Windows.Controls.TabItem"/>.
		/// The default <see cref="DefaultGridSize"/> is 3, 4.
		/// </summary>
		public GridSize DefaultGridSize
		{
			get { return (GridSize) GetValue(DefaultGridSizeProperty); }
			set
			{
				DefaultGridSize.Rows = value.Rows;
				DefaultGridSize.Columns = value.Columns;
				DefaultGridSize.Sealed = value.Sealed;
			}
		}

		#endregion Properties

		public const string RootPanelFactoryIdentifier = "rootpanel";
		public const string TitleBarFactoryIdentifier = "titlebar_factory";
		public const string TabControlFactoryIdentifier = "tabcontrol_factory";
		public const string TitleBarItemFactoryIdentifier = "titlebar_item_factory";
		public const string StatusBarFactoryIdentifier = "statusbar_factory";

		/// <summary>
		/// The <see cref="TitleBarControl"/> for the dropdowns in the title.
		/// With this property you can access every object of the dropdown.
		/// </summary>
		public TitleBarControl TitleBar { get; }

		/// <summary>
		/// The <see cref="TabControl"/> for the tabs. It allows to access each <see cref="TabUI"/>
		/// and therefore, the <see cref="System.Windows.Controls.TabItem"/>.
		/// </summary>
		public TabControlUI<SigmaWindow, TabUI> TabControl { get; set; }

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		public SigmaWindow(WPFMonitor monitor, App app, string title) : this(monitor, app, title, null)
		{

		}

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>. Since <see cref="SigmaWindow"/>
		/// heavily relies on Dragablz, a <see cref="SigmaWindow"/> has to be created at runtime.
		/// Therefore every subclass of <see cref="SigmaWindow"/> must implement exactly this constructor
		/// - otherwise, the <see cref="Dragablz.IInterTabClient"/> specified in <see cref="TabControlUI{TWindow,TTabWrapper}"/>
		/// throws an reflection exception when dragging windows out. 
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		/// <param name="other"><code>null</code> if there is no previous window - otherwise the previous window.</param>
		protected SigmaWindow(WPFMonitor monitor, App app, string title, SigmaWindow other) : base(monitor, app, title)
		{
			if (other == null)
			{
				AssignFactories(monitor);
			}

			MinHeight = 500;
			MinWidth = 750;

			FontFamily = UIResources.FontFamily;

			TitleAlignment = HorizontalAlignment.Center;

			TitleBar = CreateObjectByFactory<TitleBarControl>(TitleBarFactoryIdentifier);
			LeftWindowCommands = TitleBar;

			//TODO: add TitleBarItems via factory
			AddTitleBarItems(TitleBar/*, other*/);

			TabControl = CreateObjectByFactory<TabControlUI<SigmaWindow, TabUI>>(TabControlFactoryIdentifier);

			if (other == null)
			{
				//HACK: not Thread safe, if user is stupid and adds tabs 
				//to the registry after start (and calls this constructor via reflection)
				AddTabs(TabControl, monitor.Tabs);
			}

			Layout tabLayout = (Layout) TabControl;
			UIElement statusBar = CreateObjectByFactory<UIElement>(StatusBarFactoryIdentifier);
			DockPanel rootLayout = CreateObjectByFactory<DockPanel>(RootPanelFactoryIdentifier);

			rootLayout.Children.Add(statusBar);
			DockPanel.SetDock(statusBar, Dock.Bottom);

			rootLayout.Children.Add(tabLayout);

			Content = rootLayout;
		}

		/// <summary>
		/// Creates an object from a registry with passed key and <see cref="IMonitor"/>.
		/// </summary>
		/// <typeparam name="T">The generic type of the factory. </typeparam>
		/// <param name="identifier">The key for the registry. </param>
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreatElement"/></returns>
		protected T CreateObjectByFactory<T>(string identifier)
		{
			return CreateObjectByFactory<T>(Monitor, identifier);
		}

		/// <summary>
		/// Creates an object from a registry with passed key and the given<see cref="IMonitor"/>.
		/// </summary>
		/// <typeparam name="T">The generic type of the factory. </typeparam>
		/// <param name="monitor">The given <see cref="IMonitor"/> - the registry from the passed 
		/// monitor will be used. </param>
		/// <param name="identifier">The key for the registry. </param>
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreatElement"/></returns>
		protected T CreateObjectByFactory<T>(IMonitor monitor, string identifier)
		{
			return ((IUIFactory<T>) monitor.Registry[identifier]).CreatElement(App, this);
		}

		protected override void InitialiseComponents()
		{
			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			SetBorderBehaviour(App);

			AddResources();
		}

		/// <summary>
		/// This methods assigns the factories (if not already present) 
		/// to the registry contained in the <see cref="IMonitor"/>. 
		/// </summary>
		/// <param name="monitor"></param>
		protected virtual void AssignFactories(WPFMonitor monitor)
		{
			if (!monitor.Registry.ContainsKey(RootPanelFactoryIdentifier))
			{
				monitor.Registry[RootPanelFactoryIdentifier] = new RootPanelFactory();
			}

			if (!monitor.Registry.ContainsKey(TitleBarFactoryIdentifier))
			{
				monitor.Registry[TitleBarFactoryIdentifier] = new TitleBarFactory();
			}

			if (!monitor.Registry.ContainsKey(TabControlFactoryIdentifier))
			{
				monitor.Registry[TabControlFactoryIdentifier] = new TabControlFactory(monitor);
			}

			if (!monitor.Registry.ContainsKey(TitleBarItemFactoryIdentifier))
			{
				monitor.Registry[TitleBarItemFactoryIdentifier] = new TitleBarItemFactory();
			}

			if (!monitor.Registry.ContainsKey(StatusBarFactoryIdentifier))
			{
				monitor.Registry[StatusBarFactoryIdentifier] = new StatusBarFactory(32);
			}
		}

		/// <summary>
		/// Define how the border of the application behaves.
		/// </summary>
		/// <param name="app">The app environment. </param>
		protected virtual void SetBorderBehaviour(App app)
		{
			//This can only be set in the constructor or on start
			BorderThickness = new Thickness(1);
			BorderBrush = UIResources.AccentColorBrush;
			GlowBrush = UIResources.AccentColorBrush;

			//Disable that the title bar will get grey if not focused. 
			//And any other changes that may occur when the window is not focused.
			NonActiveWindowTitleBrush = UIResources.AccentColorBrush;
			NonActiveBorderBrush = BorderBrush;
			NonActiveGlowBrush = GlowBrush;
		}

		/// <summary>
		/// This function adds all required resources. 
		/// </summary>
		protected virtual void AddResources()
		{

		}

		/// <summary>
		/// Add specified <see cref="TitleBarItem"/>s to a given <see cref="TitleBarControl"/>.
		/// </summary>
		/// <param name="titleBarControl">The specified <see cref="TitleBarControl"/>.</param>
		private void AddTitleBarItems(TitleBarControl titleBarControl/*, SigmaWindow other*/)
		{
			//if (other == null)
			{
				titleBarControl.AddItem(new TitleBarItem("Environment", "Load", "Store",
					new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));
				titleBarControl.AddItem(new TitleBarItem("Settings", "Toggle Dark", (Action) (() => Monitor.ColourManager.Dark = !Monitor.ColourManager.Dark), "Toggle Alternate", (Action) (() => Monitor.ColourManager.Alternate = !Monitor.ColourManager.Alternate)));
				titleBarControl.AddItem(new TitleBarItem("About", "Sigma"));
			}
			//else
			{
				//TODO: copy from other?
				//load from file?
				//set this function? probably the best
				// --SetHowToAddTitleBar(...)
				//GenerateTitleBarAsUserSpecified.Invoke();
			}
		}

		/// <summary>
		/// Adds the tabs to the given <see cref="TabControlUI{T, U}"/>.
		/// </summary>
		/// <param name="tabControl">The <see cref="TabControlUI{T, U}"/>, where the <see cref="System.Windows.Controls.TabItem"/>s will be added to.</param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		protected virtual void AddTabs(TabControlUI<SigmaWindow, TabUI> tabControl, List<string> names)
		{
			foreach (string name in names)
			{
				tabControl.AddTab(name, new TabUI(name, DefaultGridSize));
			}
		}

		public override void HandleUnhandledException(object sender, UnhandledExceptionEventArgs e)
		{
			Exception exception = (Exception) e.ExceptionObject;
			this.ShowMessageAsync($"An unexpected error in {exception.Source} occurred!", exception.Message);
		}
	}
}
