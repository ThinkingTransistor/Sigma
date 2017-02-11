/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Forms;
using System.Windows.Media.Imaging;
using Dragablz.Dockablz;
using log4net.Appender;
using log4net.Core;
using MahApps.Metro.Controls.Dialogs;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;
using Sigma.Core.Monitors.WPF.ViewModel.TitleBar;
using Sigma.Core.Utils;
using Application = System.Windows.Application;
using CharacterCasing = System.Windows.Controls.CharacterCasing;
using HorizontalAlignment = System.Windows.HorizontalAlignment;
using MenuItem = System.Windows.Forms.MenuItem;
using Panel = System.Windows.Controls.Panel;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	/// <summary>
	///	The default implementation of a window that enables the following features:
	///		
	///	* a menu
	/// 
	///	* tab support
	///	
	/// * grid support
	/// 
	///	* a legend
	/// 
	///	* background running
	/// </summary>
	public class SigmaWindow : WPFWindow, IAppender, IDisposable
	{
		public const string RootPanelFactoryIdentifier = "rootpanel_factory";
		public const string TitleBarFactoryIdentifier = "titlebar_factory";
		public const string TabControlFactoryIdentifier = "tabcontrol_factory";
		public const string StatusBarFactoryIdentifier = "statusbar_factory";
		public const string LoadingIndicatorFactoryIdentifier = "loading_indicator_factory";
		public const string NotifyIconFactoryIdentifier = "notifyicon_factory";

		/// <summary>
		/// The path to the sigma icon.
		/// 
		/// This path is baked in for a reason - use our icon if you want to make us happy :)
		/// </summary>
		public const string SigmaIconPath = "pack://application:,,,/Sigma.Core.Monitors.WPF;component/Resources/icons/sigma.ico";

		/// <summary>
		/// Set a registry entry with false in the root monitor in order to do not set the icon
		/// </summary>
		public const string SigmaIconIdentifier = "sigma_icon";

		/// <summary>
		/// The children of the current window. Those
		/// may not be active anymore. Check for <see cref="IsAlive" /> -
		/// maintain the list.
		/// </summary>
		protected List<SigmaWindow> Children;

		/// <summary>
		///	The children of the window as <see cref="IReadOnlyCollection{T}"/>.
		/// </summary>
		public IReadOnlyCollection<SigmaWindow> ChildrenReadOnly => Children.AsReadOnly();

		/// <summary>
		/// The parent window of this <see cref="SigmaWindow" />.
		/// May be <c>null</c> if root window. (Check <see cref="IsRoot"/>)
		/// </summary>
		public SigmaWindow ParentWindow { get; protected set; }

		#region UIElements

		/// <summary>
		/// The UIElement that contains the <see cref="LoadingIndicatorElement"/>
		/// and the <see cref="RootContentElement"/> with different z-indexes.
		/// </summary>
		protected Grid RootElement;

		/// <summary>
		/// The element that contains the content.
		/// </summary>
		protected UIElement RootContentElement;

		/// <summary>
		/// The <see cref="TitleBarControl" /> for the dropdowns in the title.
		/// With this property you can access every object of the dropdown.
		/// </summary>
		private readonly TitleBarControl _titleBar;

		/// <summary>
		/// The element that will be displayed while loading. it is contained in the <see cref="RootElement"/>
		/// </summary>
		protected UIElement LoadingIndicatorElement;

		/// <summary>
		/// The <see cref="TitleBarControl" /> for the dropdowns in the title.
		/// With this property you can access every object of the dropdown.
		/// </summary>
		public TitleBarControl TitleBar => _titleBar;

		/// <summary>
		/// The <see cref="TabControl" /> for the tabs. It allows to access each <see cref="TabUI" />
		/// and therefore, the <see cref="TabItem" />.
		/// </summary>
		public TabControlUI<SigmaWindow, TabUI> TabControl { get; set; }

		/// <summary>
		///	The notify icon for the WPF window - if multiple windows are active (tabs teared out), they will all have the same reference. 
		/// </summary>
		public NotifyIcon NotifyIcon { get; }

		/// <summary>
		/// The <see cref="DialogHost"/> in order to show popups.
		/// It is identified with <see cref="DialogHostIdentifier"/>.
		/// </summary>
		public DialogHost DialogHost { get; private set; }

		#endregion UIElements

		/// <summary>
		/// The prefix-identifier for <see cref="DialogHost"/>.
		/// Should be unique
		/// </summary>
		private const string BaseDialogHostIdentifier = "SigmaRootDialog";

		/// <summary>
		/// The identifier for the <see cref="DialogHost"/> of this window. 
		/// </summary>
		public readonly string DialogHostIdentifier;

		/// <summary>
		///	This boolean is used to identify whether the window-behaviour to show an initialisation bar has been manually overridden.
		/// </summary>
		private bool _manualOverride;

		/// <summary>
		///	Decide whether the window is currently initialising.
		/// </summary>
		private bool _isInitializing;

		/// <summary>
		///	Set this to true while initialising the windows and false when finished in order to show a fancy indicator.
		/// </summary>
		public override bool IsInitializing
		{
			get { return _isInitializing; }
			set
			{
				if (_isInitializing != value)
				{
					_manualOverride = true;
				}

				_isInitializing = value;

				LoadingIndicatorElement.Visibility = _isInitializing ? Visibility.Visible : Visibility.Hidden;
				TabControl.WrappedContent.Visibility = _isInitializing ? Visibility.Hidden : Visibility.Visible;
			}
		}

		/// <summary>
		/// The index of the current window.
		/// </summary>
		public long WindowIndex { get; }

		/// <summary>
		/// The count that will be used to iterate through windows.
		/// </summary>
		private static long _windowCount;

		/// <summary>
		///     Determines whether this is the root window.
		/// </summary>
		public bool IsRoot => ParentWindow == null;

		/// <summary>
		///     Determines whether this window is closed or about to close.
		/// </summary>
		public bool IsAlive { get; private set; } = true;

		/// <summary>
		///		This boolean decides, whether the program runs in background or in foreground. 
		/// </summary>
		public bool IsRunningInBackground { get; private set; }

		/// <summary>
		/// This boolean will be set to true if the application is force closed in order to check for it.
		/// </summary>
		private bool _forceClose;

		#region DependencyProperties

		public static readonly DependencyProperty DefaultGridSizeProperty = DependencyProperty.Register("DefaultGridSize",
			typeof(GridSize), typeof(WPFWindow), new UIPropertyMetadata(new GridSize(3, 4)));

		#endregion DependencyProperties

		#region Properties

		/// <summary>
		///     The DefaultGridSize for each newly created <see cref="TabItem" />.
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
		///     The constructor for the <see cref="WPFWindow" />.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="System.Windows.Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		public SigmaWindow(WPFMonitor monitor, Application app, string title) : this(monitor, app, title, null) { }

		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />. Since <see cref="SigmaWindow" />
		///     heavily relies on Dragablz, a <see cref="SigmaWindow" /> has to be created at runtime.
		///     Therefore every subclass of <see cref="SigmaWindow" /> must implement exactly this constructor
		///     - otherwise, the <see cref="Dragablz.IInterTabClient" /> specified in
		///     <see cref="TabControlUI{TWindow,TTabWrapper}" />
		///     throws a reflection exception when dragging windows out.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		/// <param name="other"><code>null</code> if there is no previous window - otherwise the previous window.</param>
		protected SigmaWindow(WPFMonitor monitor, Application app, string title, SigmaWindow other) : base(monitor, app, title)
		{
			WindowIndex = _windowCount++;

			SetIcon(monitor);

			Children = new List<SigmaWindow>();

			if (other == null)
			{
				AssignFactories(monitor.Registry, app, monitor);
			}
			else
			{
				SetParent(other);
			}

			InitialiseDefaultValues();

			RootElement = new Grid();

			DialogHostIdentifier = BaseDialogHostIdentifier + WindowIndex;
			DialogHost = new DialogHost { Identifier = DialogHostIdentifier };
			LoadingIndicatorElement = CreateObjectByFactory<UIElement>(LoadingIndicatorFactoryIdentifier);
			RootContentElement = CreateContent(monitor, other, out _titleBar);

			RootElement.Children.Add(RootContentElement);
			RootElement.Children.Add(DialogHost);
			RootElement.Children.Add(LoadingIndicatorElement);

			if (other == null)
			{
				NotifyIcon = CreateObjectByFactory<NotifyIcon>(NotifyIconFactoryIdentifier);
			}
			else
			{
				LoadingIndicatorElement.Visibility = Visibility.Hidden;
				NotifyIcon = other.NotifyIcon;
			}

			Content = RootElement;

			App.Startup += OnStart;
			Closing += OnClosing;
			Closed += OnClosed;
		}

		#region Lifecycle
		protected virtual void OnStart(object sender, StartupEventArgs startupEventArgs)
		{
			if (!_manualOverride)
			{
				IsInitializing = false;
			}
		}

		protected virtual void CustomMinimise()
		{
			IsRunningInBackground = true;

			WindowState = WindowState.Minimized;
			Hide();
		}

		protected virtual void CustomMaximise()
		{
			if (IsRunningInBackground)
			{
				IsRunningInBackground = false;

				Show();
				WindowState = WindowState.Normal;
			}
		}

		private void OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			if (!_forceClose && !IsRunningInBackground)
			{
				// if all other tabs are closed
				if (IsRoot && Children.Count == 0)
				{
					Debug.WriteLine("Minimising");
					CustomMinimise();

					e.Cancel = true;
				}
			}
		}

		protected virtual void OnClosed(object sender, EventArgs eventArgs)
		{
			App.Startup -= OnStart;
			Closing -= OnClosing;
			Closed -= OnClosed;

			IsAlive = false;
			Dispose();
		}

		/// <summary>
		/// 	Since Sigma can run in background, a method is required that force closes the application. 
		/// </summary>
		protected virtual void ForceClose()
		{
			_forceClose = true;
			Close();
		}

		#endregion Lifecycle

		#region FactoryPattern
		/// <summary>
		///     Creates an object from a registry with passed key and <see cref="IMonitor" />.
		/// </summary>
		/// <typeparam name="T">The generic type of the factory. </typeparam>
		/// <param name="identifier">The key for the registry. </param>
		/// <param name="parameters">The parameters to create the element - in most cases this can be left empty.</param>
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreateElement" /></returns>
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
		/// <returns>The newly created object. (<see cref="IUIFactory{T}.CreateElement" /></returns>
		protected T CreateObjectByFactory<T>(IMonitor monitor, string identifier, params object[] parameters)
		{
			return ((IUIFactory<T>) monitor.Registry[identifier]).CreateElement(App, this, parameters);
		}

		/// <summary>
		///     This methods assigns the factories (if not already present)
		///     to the registry passed. This is used to ensure that default factories are assigned. 
		/// </summary>
		protected virtual void AssignFactories(IRegistry registry, Application app, WPFMonitor monitor)
		{
			if (!registry.ContainsKey(RootPanelFactoryIdentifier))
			{
				registry[RootPanelFactoryIdentifier] = new RootPanelFactory();
			}

			if (!registry.ContainsKey(TitleBarFactoryIdentifier))
			{
				registry[TitleBarFactoryIdentifier] = new TitleBarFactory(registry);
			}

			if (!registry.ContainsKey(TabControlFactoryIdentifier))
			{
				registry[TabControlFactoryIdentifier] = new TabControlFactory(monitor);
			}

			if (!registry.ContainsKey(StatusBarFactoryIdentifier))
			{
				registry[StatusBarFactoryIdentifier] = new StatusBarFactory(registry, 32,
					new GridLength(3, GridUnitType.Star), new GridLength(1, GridUnitType.Star), new GridLength(3, GridUnitType.Star));
			}

			if (!registry.ContainsKey(LoadingIndicatorFactoryIdentifier))
			{
				registry[LoadingIndicatorFactoryIdentifier] = new LoadingIndicatorFactory();
			}

			if (!registry.ContainsKey(NotifyIconFactoryIdentifier))
			{
				MenuItem[] items = new MenuItem[2];

				//TODO: localise
				//TODO: find correct place in code for this part
				items[0] = new MenuItem("Open") { DefaultItem = true };
				items[0].Click += (sender, args) => ExecuteOnRoot(monitor, window => window.CustomMaximise());

				items[1] = new MenuItem("Exit");
				items[1].Click += (sender, args) => ExecuteOnRoot(monitor, window => window.PropagateAction(win => win.ForceClose()));

				//TODO: HACK: path not dynamic
				registry[NotifyIconFactoryIdentifier] = new SigmaNotifyIconFactory("Sigma", @"C:\Users\Plainer\Dropbox\!school\5AHIT\Diplomarbeit\Logo\export\sigma.ico", (sender, args) => ExecuteOnRoot(monitor, window => window.CustomMaximise()), items);
			}
		}

		#endregion FactoryPattern

		#region UICreation

		/// <summary>
		///     Initialise default layout configuration.
		/// </summary>
		private void InitialiseDefaultValues()
		{
			MinHeight = 650;
			MinWidth = 880;
			FontFamily = UIResources.FontFamily;
			TitleAlignment = HorizontalAlignment.Center;
		}

		/// <summary>
		///     This function adds all required resources.
		///		Override as you wish. 
		/// </summary>
		protected virtual void AddResources() { }

		/// <summary>
		///		This method initialises the border (<see cref="SetBorderBehaviour"/>) and adds required resources (<see cref="AddResources"/>)
		/// </summary>
		protected override void InitialiseComponents()
		{
			//TODO: style file?
			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			SetBorderBehaviour(App);

			AddResources();
		}

		/// <summary>
		///     Define how the border of the application behaves.
		/// </summary>
		/// <param name="app">The app environment. </param>
		protected virtual void SetBorderBehaviour(Application app)
		{
			//TODO: style? 

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
		///		Set the icon for a window (based on a registry entry in the monitor). 
		///		If <see cref="SigmaIconIdentifier"/> is set to false, this method does not do anything. 
		/// </summary>
		/// <param name="monitor"></param>
		protected virtual void SetIcon(WPFMonitor monitor)
		{
			bool useIcon;

			// if not set or manually set
			if (!monitor.Registry.TryGetValue(SigmaIconIdentifier, out useIcon) || useIcon)
			{
				Icon = new BitmapImage(new Uri(SigmaIconPath));
			}
		}

		/// <summary>
		///		This method creates the root layout for the window and automatically adds the items that should be in the root panel (i.e. references are required).
		/// </summary>
		/// <param name="monitor">The monitor from which the legends are copied. </param>
		/// <param name="other">The previous <see cref="SigmaWindow"/> - some values propagate automatically to the new window (e.g. tabs).</param>
		/// <param name="titleBarControl">The <see cref="TitleBarControl"/>, since it belongs to the window itself (not root content).</param>
		/// <returns>The newly created root panel. (In this case where the tabs etc. are placed)</returns>
		protected virtual Panel CreateContent(WPFMonitor monitor, SigmaWindow other, out TitleBarControl titleBarControl)
		{
			titleBarControl = CreateObjectByFactory<TitleBarControl>(TitleBarFactoryIdentifier);
			LeftWindowCommands = TitleBar;

			TabControl = CreateObjectByFactory<TabControlUI<SigmaWindow, TabUI>>(TabControlFactoryIdentifier);

			if (other == null)
			{
				AddTabs(TabControl, monitor.Tabs);
			}

			Layout tabLayout = (Layout) TabControl;
			// ReSharper disable once CoVariantArrayConversion
			UIElement statusBar = CreateObjectByFactory<UIElement>(StatusBarFactoryIdentifier, monitor.Legends.Values.ToArray());
			DockPanel rootLayout = CreateObjectByFactory<DockPanel>(RootPanelFactoryIdentifier);

			rootLayout.Children.Add(statusBar);
			DockPanel.SetDock(statusBar, Dock.Bottom);

			rootLayout.Children.Add(tabLayout);
			return rootLayout;
		}

		/// <summary>
		///     Adds the tabs to the given <see cref="TabControlUI{T, U}" />.
		/// </summary>
		/// <param name="tabControl">
		///     The <see cref="TabControlUI{T, U}" />, where the
		///     <see cref="TabItem" />s will be added to.
		/// </param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		protected virtual void AddTabs(TabControlUI<SigmaWindow, TabUI> tabControl, List<string> names)
		{
			foreach (string name in names)
			{
				tabControl.AddTab(name, new TabUI(name, DefaultGridSize));
			}
		}

		#endregion UICreation

		#region CommandExecution

		/// <summary>
		///     Finds the root of a given <see cref="SigmaWindow" />
		/// </summary>
		/// <param name="start">The point to begin the search. May not be null. </param>
		/// <returns>The root window. </returns>
		private static SigmaWindow FindRoot(SigmaWindow start)
		{
			if (start == null)
			{
				throw new ArgumentNullException(nameof(start));
			}

			while (start.ParentWindow != null)
			{
				start = start.ParentWindow;
			}

			return start;
		}

		/// <summary>
		///     Execute an action on every active <see cref="SigmaWindow" />.
		///     This method also maintains the children reference list. (<see cref="Children" />)
		/// </summary>
		/// <param name="action">The <see cref="Action" /> that will be executed. </param>
		public virtual void PropagateAction(Action<SigmaWindow> action)
		{
			PropagateActionDownwards(action, FindRoot(this));
		}

		/// <summary>
		/// Execute an action statically on a window.
		/// </summary>
		/// <param name="window">The window the <see cref="Action"/> will be executed on.</param>
		/// <param name="action">The <see cref="Action"/> that will be executed.</param>
		protected static void ExecuteOnWindow(SigmaWindow window, Action<SigmaWindow> action)
		{
			action(window);
		}

		/// <summary>
		/// Execute an action statically on the root window.
		/// 
		/// Warning: this only works if the <see cref="WPFMonitor.Window"/> has been assigned and maintained properly. 
		/// </summary>
		/// <param name="monitor">The <see cref="WPFMonitor"/> to receive the main window from. </param>
		/// <param name="action">The <see cref="Action"/> that will be executed.</param>
		protected static void ExecuteOnRoot(WPFMonitor monitor, Action<SigmaWindow> action)
		{
			ExecuteOnWindow((SigmaWindow) monitor.Window, action);
		}

		/// <summary>
		/// Execute an action on the element <see cref="start" /> and all children of start.
		/// This method also maintains the children reference list. (<see cref="Children" />)
		/// </summary>
		/// <param name="action">The <see cref="Action" /> that will be executed. </param>
		/// <param name="start">The element to begin (normally the root and then internally recursive). </param>
		private static void PropagateActionDownwards(Action<SigmaWindow> action, SigmaWindow start)
		{
			action(start);

			for (int i = 0; i < start.Children.Count; i++)
			{
				if (start.Children[i].IsAlive)
				{
					PropagateActionDownwards(action, start.Children[i]);
				}
				else
				{
					start.Children.RemoveAt(i--);
				}
			}
		}

		#endregion CommandExecution

		/// <summary>
		/// The method that handles all unhandled exception (log + popup). 
		/// </summary>
		/// <param name="sender">The sender of the event.</param>
		/// <param name="e">The args for the unhandled exception.</param>
		public override void HandleUnhandledException(object sender, UnhandledExceptionEventArgs e)
		{
			Exception exception = (Exception) e.ExceptionObject;
			this.ShowMessageAsync($"An unexpected error in {exception.Source} occurred!", exception.Message);
		}

		/// <summary>
		///		Set this window to be a child of another window.
		/// </summary>
		/// <param name="parent"></param>
		protected virtual void SetParent(SigmaWindow parent)
		{
			if (ReferenceEquals(ParentWindow, parent)) return;

			// remove from parent (if there is any)
			ParentWindow?.Children.Remove(this);

			// add to new parent
			ParentWindow = parent;

			if (parent == null)
			{
				App.MainWindow = this;
				Monitor.Window = this;
			}
			else
			{
				parent.Children.Add(this);
			}
		}

		/// <summary>Returns the string representation of a <see cref="T:System.Windows.Controls.Control" /> object. </summary>
		/// <returns>A string that represents the control.</returns>
		public override string ToString()
		{
			return $"SigmaWindow[{WindowIndex}]";
		}

		#region IDisposable
		public void Dispose()
		{
			if (ParentWindow == null)
			{
				if (Children.Count > 0)
				{
					SigmaWindow newRoot = Children[0];
					// set one child as parent
					newRoot.SetParent(null);

					// set all others children parent
					for (int i = 1; i < Children.Count; i++)
					{
						Children[i].SetParent(newRoot);
					}

					App.MainWindow = newRoot;
				}
			}
			else
			{
				foreach (SigmaWindow child in Children)
				{
					child.SetParent(ParentWindow);
				}
			}

			SetParent(null);

			ParentWindow = null;
			RootElement = null;
			RootContentElement = null;
			LoadingIndicatorElement = null;
			TabControl = null;
			DialogHost = null;
		}

		~SigmaWindow()
		{
			Dispose();
		}

		#endregion

		#region Logging

		/// <summary>Log the logging event in Appender specific way.</summary>
		/// <param name="loggingEvent">The event to log</param>
		/// <remarks>
		/// <para>
		/// This method is called to log a message into this appender.
		/// </para>
		/// </remarks>
		public override void DoAppend(LoggingEvent loggingEvent)
		{
			ToolTipIcon toolTip = ToolTipIcon.Info;

			if (loggingEvent.Level.Value < Level.Warn.Value)
			{
				toolTip = ToolTipIcon.Error;
			}
			else if (loggingEvent.Level.Value < Level.Info.Value)
			{
				toolTip = ToolTipIcon.Warning;
			}

			NotifyIcon.ShowBalloonTip(2000, "Warning", loggingEvent.MessageObject.ToString(), toolTip);
		}

		#endregion
	}
}