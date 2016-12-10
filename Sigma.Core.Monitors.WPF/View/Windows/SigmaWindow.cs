/* 
MIT License

Copyright (c) 2016 Florian C�sar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using Dragablz.Dockablz;
using MahApps.Metro.Controls.Dialogs;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;
using Sigma.Core.Utils;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	public class SigmaWindow : WPFWindow
	{
		public const string RootPanelFactoryIdentifier = "rootpanel_factory";
		public const string TitleBarFactoryIdentifier = "titlebar_factory";
		public const string TabControlFactoryIdentifier = "tabcontrol_factory";
		public const string TitleBarItemFactoryIdentifier = "titlebar_item_factory";
		public const string StatusBarFactoryIdentifier = "statusbar_factory";

		#region DependencyProperties

		public static readonly DependencyProperty DefaultGridSizeProperty = DependencyProperty.Register("DefaultGridSize",
			typeof(GridSize), typeof(WPFWindow), new UIPropertyMetadata(new GridSize(3, 4)));

		#endregion DependencyProperties

		/// <summary>
		///     The <see cref="TitleBarControl" /> for the dropdowns in the title.
		///     With this property you can access every object of the dropdown.
		/// </summary>
		private readonly TitleBarControl _titleBar;

		/// <summary>
		///     Determines whether the window has been closed.
		/// </summary>
		private bool _isAlive = true;

		/// <summary>
		///     The children of the current window. Those
		///     may not be active anymore. Check for <see cref="IsAlive" /> -
		///     maintain the list.
		/// </summary>
		protected List<SigmaWindow> Children;

		/// <summary>
		///     The parent window of this <see cref="SigmaWindow" />.
		///     May be <c>null</c> if root window.
		/// </summary>
		protected SigmaWindow ParentWindow;

		/// <summary>
		///     Reference to the root <see cref="SigmaWindow" />.
		///     This value will never be <c>null</c>.
		/// </summary>
		protected SigmaWindow RootWindow;

		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		public SigmaWindow(WPFMonitor monitor, App app, string title) : this(monitor, app, title, null)
		{
		}

		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />. Since <see cref="SigmaWindow" />
		///     heavily relies on Dragablz, a <see cref="SigmaWindow" /> has to be created at runtime.
		///     Therefore every subclass of <see cref="SigmaWindow" /> must implement exactly this constructor
		///     - otherwise, the <see cref="Dragablz.IInterTabClient" /> specified in
		///     <see cref="TabControlUI{TWindow,TTabWrapper}" />
		///     throws an reflection exception when dragging windows out.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		/// <param name="other"><code>null</code> if there is no previous window - otherwise the previous window.</param>
		protected SigmaWindow(WPFMonitor monitor, App app, string title, SigmaWindow other) : base(monitor, app, title)
		{
			ParentWindow = other;
			RootWindow = FindRoot(this);
			Children = new List<SigmaWindow>();

			if (other == null)
				AssignFactories(monitor.Registry, app, monitor);
			else
				other.Children.Add(this);

			Closed += (sender, args) => _isAlive = false;

			InitialiseDefaultValues();


			Panel rootLayout = CreateContent(monitor, other, out _titleBar);

			Content = rootLayout;
		}

		#region Properties

		/// <summary>
		///     The DefaultGridSize for each newly created <see cref="System.Windows.Controls.TabItem" />.
		///     The default <see cref="DefaultGridSize" /> is 3, 4.
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

		/// <summary>
		///     The <see cref="TitleBarControl" /> for the dropdowns in the title.
		///     With this property you can access every object of the dropdown.
		/// </summary>
		public TitleBarControl TitleBar => _titleBar;

		/// <summary>
		///     The <see cref="TabControl" /> for the tabs. It allows to access each <see cref="TabUI" />
		///     and therefore, the <see cref="System.Windows.Controls.TabItem" />.
		/// </summary>
		public TabControlUI<SigmaWindow, TabUI> TabControl { get; set; }

		/// <summary>
		///     Determines whether this is the root window.
		/// </summary>
		public bool IsRoot => ParentWindow == null;

		/// <summary>
		///     Determines whether this window is closed or about to close.
		/// </summary>
		public bool IsAlive => _isAlive;

		protected virtual Panel CreateContent(WPFMonitor monitor, SigmaWindow other, out TitleBarControl titleBarControl)
		{
			titleBarControl = CreateObjectByFactory<TitleBarControl>(TitleBarFactoryIdentifier);
			LeftWindowCommands = TitleBar;

			//TODO: add TitleBarItems via factory
			AddTitleBarItems(TitleBar /*, other*/);

			TabControl = CreateObjectByFactory<TabControlUI<SigmaWindow, TabUI>>(TabControlFactoryIdentifier);

			if (other == null)
				AddTabs(TabControl, monitor.Tabs);

			Layout tabLayout = (Layout) TabControl;
			// ReSharper disable once CoVariantArrayConversion
			UIElement statusBar = CreateObjectByFactory<UIElement>(StatusBarFactoryIdentifier, monitor.Legends.ToArray());
			DockPanel rootLayout = CreateObjectByFactory<DockPanel>(RootPanelFactoryIdentifier);

			rootLayout.Children.Add(statusBar);
			DockPanel.SetDock(statusBar, Dock.Bottom);

			rootLayout.Children.Add(tabLayout);
			return rootLayout;
		}

		/// <summary>
		///     Initialise default layout configuration.
		/// </summary>
		private void InitialiseDefaultValues()
		{
			MinHeight = 500;
			MinWidth = 750;
			FontFamily = UIResources.FontFamily;
			TitleAlignment = HorizontalAlignment.Center;
		}


		/// <summary>
		///     Finds the root of a given <see cref="SigmaWindow" />
		/// </summary>
		/// <param name="start">The point to begin the search. May not be null. </param>
		/// <returns>The root window. </returns>
		private static SigmaWindow FindRoot(SigmaWindow start)
		{
			if (start == null) throw new ArgumentNullException(nameof(start));

			while (start.ParentWindow != null)
				start = start.ParentWindow;

			return start;
		}

		/// <summary>
		///     Creates an object from a registry with passed key and <see cref="IMonitor" />.
		/// </summary>
		/// <typeparam name="T">The generic type of the factory. </typeparam>
		/// <param name="identifier">The key for the registry. </param>
		/// <param name="parameters">The parameters to create the element - in most cases this can be left empty.</param>
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreatElement" /></returns>
		protected T CreateObjectByFactory<T>(string identifier, params object[] parameters)
		{
			return CreateObjectByFactory<T>(Monitor, identifier, parameters);
		}

		/// <summary>
		///     Creates an object from a registry with passed key and the given<see cref="IMonitor" />.
		/// </summary>
		/// <typeparam name="T">The generic type of the factory. </typeparam>
		/// <param name="monitor">
		///     The given <see cref="IMonitor" /> - the registry from the passed
		///     monitor will be used.
		/// </param>
		/// <param name="identifier">The key for the registry. </param>
		/// <param name="parameters">The parameters to create the element - in most cases this can be left empty.</param>
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreatElement" /></returns>
		protected T CreateObjectByFactory<T>(IMonitor monitor, string identifier, params object[] parameters)
		{
			return ((IUIFactory<T>) monitor.Registry[identifier]).CreatElement(App, this, parameters);
		}

		protected override void InitialiseComponents()
		{
			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			SetBorderBehaviour(App);

			AddResources();
		}

		/// <summary>
		///     This methods assigns the factories (if not already present)
		///     to the registry passed.
		/// </summary>
		protected virtual void AssignFactories(IRegistry registry, App app, WPFMonitor monitor)
		{
			if (!registry.ContainsKey(RootPanelFactoryIdentifier))
				registry[RootPanelFactoryIdentifier] = new RootPanelFactory();

			if (!registry.ContainsKey(TitleBarFactoryIdentifier))
				registry[TitleBarFactoryIdentifier] = new TitleBarFactory();

			if (!registry.ContainsKey(TabControlFactoryIdentifier))
				registry[TabControlFactoryIdentifier] = new TabControlFactory(monitor);

			if (!registry.ContainsKey(TitleBarItemFactoryIdentifier))
				registry[TitleBarItemFactoryIdentifier] = new TitleBarItemFactory();

			if (!registry.ContainsKey(StatusBarFactoryIdentifier))
				registry[StatusBarFactoryIdentifier] = new StatusBarFactory(new Registry(registry), 32, 0, new GridLength());
		}

		/// <summary>
		///     Define how the border of the application behaves.
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
		///     This function adds all required resources.
		/// </summary>
		protected virtual void AddResources()
		{
		}

		/// <summary>
		///     Add specified <see cref="TitleBarItem" />s to a given <see cref="TitleBarControl" />.
		/// </summary>
		/// <param name="titleBarControl">The specified <see cref="TitleBarControl" />.</param>
		private void AddTitleBarItems(TitleBarControl titleBarControl /*, SigmaWindow other*/)
		{
			//if (other == null)
			{
				titleBarControl.AddItem(new TitleBarItem("Environment", "Load", "Store",
					new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));
				titleBarControl.AddItem(new TitleBarItem("Settings", "Toggle Dark",
					(Action) (() => Monitor.ColourManager.Dark = !Monitor.ColourManager.Dark), "Toggle Alternate",
					(Action) (() => Monitor.ColourManager.Alternate = !Monitor.ColourManager.Alternate)));
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
		///     Adds the tabs to the given <see cref="TabControlUI{T, U}" />.
		/// </summary>
		/// <param name="tabControl">
		///     The <see cref="TabControlUI{T, U}" />, where the
		///     <see cref="System.Windows.Controls.TabItem" />s will be added to.
		/// </param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		protected virtual void AddTabs(TabControlUI<SigmaWindow, TabUI> tabControl, List<string> names)
		{
			foreach (string name in names)
				tabControl.AddTab(name, new TabUI(name, DefaultGridSize));
		}

		/// <summary>
		///     Execute an action on every active <see cref="SigmaWindow" />.
		///     This method also maintains the children reference list. (<see cref="Children" />)
		/// </summary>
		/// <param name="action">The <see cref="Action" /> that will be executed. </param>
		public virtual void PropagateAction(Action<SigmaWindow> action)
		{
			PropagateActionDownwards(action, RootWindow);
		}

		/// <summary>
		///     Execute an action on the element <see cref="start" /> and all children of start.
		///     This method also maintains the children reference list. (<see cref="Children" />)
		/// </summary>
		/// <param name="action">The <see cref="Action" /> that will be executed. </param>
		/// <param name="start">The element to begin (normally the root and then internally recursive). </param>
		private static void PropagateActionDownwards(Action<SigmaWindow> action, SigmaWindow start)
		{
			action(start);

			for (int i = 0; i < start.Children.Count; i++)
				if (start.Children[i].IsAlive)
					PropagateActionDownwards(action, start.Children[i]);
				else
					start.Children.RemoveAt(i--);
		}

		public override void HandleUnhandledException(object sender, UnhandledExceptionEventArgs e)
		{
			Exception exception = (Exception) e.ExceptionObject;
			this.ShowMessageAsync($"An unexpected error in {exception.Source} occurred!", exception.Message);
		}
	}
}