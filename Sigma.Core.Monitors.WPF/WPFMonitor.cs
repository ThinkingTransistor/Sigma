/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Threading;
using log4net;
using log4net.Appender;
using log4net.Core;
using log4net.Filter;
using log4net.Repository.Hierarchy;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.View.Colours;
using Sigma.Core.Monitors.WPF.View.Themes;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF
{
	/// <summary>
	/// This <see cref="IMonitor" /> is the default visualisation monitor shipped with the big Sigma.
	/// The <see cref="WPFMonitor" /> is designed to run on the Windows platform.
	/// </summary>
	// ReSharper disable once InconsistentNaming
	public class WPFMonitor : MonitorAdapter, IAppender
	{
		/// <summary>
		/// The logger.
		/// </summary>
		private readonly ILog _log = LogManager.GetLogger(typeof(WPFMonitor));

		/// <summary>
		/// The type of the window that will be created.
		/// A type is passed in order to prevent generics.
		/// (Mostly that user does not have to care about it)
		/// </summary>
		private readonly Type _windowType;

		/// <summary>
		/// The root application for all WPF interactions.
		/// </summary>
		private App _app;

		/// <summary>
		/// This property returns the current window.
		/// <see cref="Window" /> is <see langword="null" /> until <see cref="SigmaEnvironment.Prepare" /> has been called.
		/// </summary>
		public WPFWindow Window { get; set; }

		/// <summary>
		/// The title of the window.
		/// </summary>
		private string _title;

		/// <summary>
		/// Property for the title of the window.
		///		The title of the corresponding window is also set. 
		/// </summary>
		public string Title
		{
			get { return _title; }
			set
			{
				_title = value;

				Window?.Dispatcher.Invoke(() => Window.Title = _title);
			}
		}

		/// <summary>
		/// The list of tabs that are available. These have to be set <b>before</b> <see cref="SigmaEnvironment.Prepare" />.
		///		Tabs can easily be added via <see cref="AddTabs"/> - see the documentation for additional informations on how to add tabs after prepare. 
		/// </summary>
		public List<string> Tabs { get; private set; }

		/// <summary>
		///		The internal mapping between a given string (displayed name of the status bar) and the actual <see cref="StatusBarLegendInfo"/>.
		///		This is mainly used for <see cref="GetLegendInfo"/>.
		/// </summary>
		internal Dictionary<string, StatusBarLegendInfo> Legends { get; private set; }

		/// <summary>
		/// The <see cref="IColourManager" /> to control the look and feel of the application.
		/// </summary>
		public IColourManager ColourManager { get; }

		/// <summary>
		/// Actions assigned to this list will listen
		/// to the onStart-event of the <see cref="_app" />.
		/// </summary>
		private readonly List<Action<object>> _onWindowStartup = new List<Action<object>>();

		/// <summary>
		/// This <see cref="bool" /> will be set to true,
		/// as soon as all <see cref="_onWindowStartup" />
		/// actions have been executed.
		/// </summary>
		private bool _onWindowStartupExecuted;

		/// <summary>
		/// The thread in which the UI runs.
		/// </summary>
		private Thread _wpfThread;

		/// <summary>
		/// The Priority with which the thread will be started.
		/// The default is <see cref="ThreadPriority.Normal" />.
		/// </summary>
		public ThreadPriority Priority { get; set; } = ThreadPriority.Normal;

		/// <summary>
		/// The UI culture info for all windows. Changes to this variable do not update if currently running. Use <see cref="UiCultureInfo"/> instead.
		/// </summary>
		private CultureInfo _uiCultureInfo;

		/// <summary>
		/// The UI culture info for all windows. Changes can be made at runtime but some UI elements may not update. (Hopefully in the future it automatically updates)
		/// </summary>
		public CultureInfo UiCultureInfo
		{
			get { return _uiCultureInfo; }
			set
			{
				_uiCultureInfo = value;

				_log.Info($"UI language changed to {_uiCultureInfo}");

				if (_wpfThread != null)
				{
					_wpfThread.CurrentUICulture = _uiCultureInfo;
				}
			}
		}

		/// <summary>
		///	Every new log entry has to pass through this filter - if it passes the filter,
		/// it will get through to the root window (<see cref="Window"/>), otherwise it will be discarded.
		/// The default is a <see cref="LevelRangeFilter"/> with a <see cref="LevelRangeFilter.LevelMin"/> of <see cref="Level.Warn"/>.
		/// 
		/// If set to <c>null</c>, all messages will get passed.
		/// </summary>
		public IFilter LogFilter { get; set; } = new LevelRangeFilter { LevelMin = Level.Warn };

		/// <summary>
		/// Determine whether to close the given <see cref="SigmaEnvironment"/> when the <see cref="SignalStop"/> method
		/// of the monitor has been called. <c>true</c> per default.
		/// </summary>
		public bool StopSigmaOnClose { get; set; } = true;

		#region Constructor

		/// <summary>
		/// The constructor for the WPF Monitor that relies on <see cref="SigmaWindow" /> and the current 
		///		ThreadCulture. 
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		public WPFMonitor(string title) : this(title, typeof(SigmaWindow)) { }

		/// <summary>
		/// The constructor for the WPF Monitor - it uses the current <see cref="Thread.CurrentUICulture"/>.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="window">
		/// The type of the <see cref="WPFWindow" /> that will be displayed. This window requires a constructor
		/// whit the same arguments as <see cref="WPFWindow" />.
		/// </param>
		public WPFMonitor(string title, Type window) : this(title, window, Thread.CurrentThread.CurrentUICulture) { }

		/// <summary>
		///		The constructor for the WPF Monitor with a given title and UI culture info. <see cref="SigmaWindow"/> will be used.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="uiCultureInfo">The culture info used for the UI (language).</param>
		public WPFMonitor(string title, string uiCultureInfo) : this(title, typeof(SigmaWindow), uiCultureInfo) { }

		/// <summary>
		///		The constructor for the WPF Monitor with a given title and UI culture info. <see cref="SigmaWindow"/> will be used.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="uiCultureInfo">The culture info used for the UI (language).</param>
		public WPFMonitor(string title, CultureInfo uiCultureInfo) : this(title, typeof(SigmaWindow), uiCultureInfo) { }
		/// <summary>
		/// The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="window">
		/// The type of the <see cref="WPFWindow" /> that will be displayed. This window requires a constructor
		/// whit the same arguments as <see cref="WPFWindow" />.
		/// </param>
		/// <param name="uiCultureInfo">The culture info used for the UI (language).</param>
		public WPFMonitor(string title, Type window, string uiCultureInfo) : this(title, window, CultureInfo.GetCultureInfo(uiCultureInfo)) { }

		/// <summary>
		/// The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="window">
		/// The type of the <see cref="WPFWindow" /> that will be displayed. This window requires a constructor
		/// whit the same arguments as <see cref="WPFWindow" />.
		/// </param>
		/// <param name="uiCultureInfo">The culture info used for the UI (language).</param>
		public WPFMonitor(string title, Type window, CultureInfo uiCultureInfo)
		{
			if (!window.IsSubclassOf(typeof(WPFWindow)))
			{
				throw new ArgumentException($"Type {window} does not extend from {typeof(WPFWindow)}!");
			}

			_uiCultureInfo = uiCultureInfo;

			Title = title;
			_windowType = window;

			// default colours are taken from the ColourManager itself
			ColourManager = new ColourManager();

			_log.Info($"{nameof(WPFMonitor)} has been created.");

		}

		#endregion Constructor

		/// <summary>
		/// This method adds a list of given tabs to <see cref="Tabs"/>. If you want to add a tab after <see cref="SigmaEnvironment.Prepare"/> has been called,
		/// call this method inside the window dispatcher (<see cref="WindowDispatcher"/>).
		/// </summary>
		/// <param name="tabs"></param>
		public void AddTabs(params string[] tabs)
		{
			Tabs.AddRange(tabs);

			_log.Debug($"Added {tabs.Length} tabs: {string.Join(", ", tabs)}.");
		}

		#region Legend

		/// <summary>
		///		Add a <see cref="StatusBarLegendInfo"/> to the UI. It will be (with a normal configuration) 
		///		displayed in the lower right corner - Panels can be marked with an info.
		/// </summary>
		/// <param name="legend">The info that will be added.</param>
		/// <returns>The passed legend for chaining / storing in a variable. </returns>
		public StatusBarLegendInfo AddLegend(StatusBarLegendInfo legend)
		{
			AddLegends(legend);

			return GetLegendInfo(legend.Name);
		}

		/// <summary>
		/// This method is equal to <see cref="AddLegend"/> but it adds multiple legends.
		/// </summary>
		/// <param name="legends"></param>
		public void AddLegends(params StatusBarLegendInfo[] legends)
		{
			lock (_onWindowStartup)
			{
				if (_onWindowStartupExecuted)
				{
					throw new InvalidOperationException("Window has already been started - StatusBar update not supported (yet).");
				}

				AddLegends(Legends, legends);
			}
		}

		/// <summary>
		/// Receive a given <see cref="StatusBarLegendInfo"/> defined by its name (<see cref="StatusBarLegendInfo.Name"/>).
		/// </summary>
		/// <param name="name"></param>
		/// <returns></returns>
		public StatusBarLegendInfo GetLegendInfo(string name)
		{
			return Legends[name];
		}

		/// <summary>
		/// Return all legends to be able to iterate through.
		/// </summary>
		/// <returns></returns>
		public IEnumerable<StatusBarLegendInfo> GetLegends()
		{
			return Legends.Values;
		}

		private static void AddLegends(IDictionary<string, StatusBarLegendInfo> legends, IEnumerable<StatusBarLegendInfo> legendInfo)
		{
			foreach (StatusBarLegendInfo statusBarLegendInfo in legendInfo)
			{
				legends.Add(statusBarLegendInfo.Name, statusBarLegendInfo);
			}
		}

		#endregion Legend

		#region Lifecycle

		/// <inheritdoc />
		public override void Initialise()
		{
			base.Initialise();

			Tabs = new List<string>();
			Legends = new Dictionary<string, StatusBarLegendInfo>();

			_log.Debug($"{nameof(WPFMonitor)} has been initialised.");
		}

		//TODO: implement
		//public void Reload()
		//{
		//	_app.Dispatcher.Invoke(() =>
		//	{
		//		WPFWindow oldWindow = Window;

		//		Window = (WPFWindow) Activator.CreateInstance(_windowType, this, _app, Title);
		//		ColourManager.Window = Window;

		//		_app.Dispatcher.Invoke(() => _app.Run(Window));

		//		oldWindow.Close();
		//	});
		//}

		/// <inheritdoc />
		public override void Start()
		{
			ThreadUtils.BlockingThread wpfThread = new ThreadUtils.BlockingThread(reset =>
			{
				_app = new App(this);
				ColourManager.App = _app;

				Window = (WPFWindow) Activator.CreateInstance(_windowType, this, _app, _title);
				AssignToLog();

				AppDomain.CurrentDomain.UnhandledException += Window.HandleUnhandledException;

				ColourManager.Window = Window;

				lock (_onWindowStartup)
				{
					if (_onWindowStartup != null)
					{
						foreach (Action<object> action in _onWindowStartup)
						{
							_app.Startup += (sender, args) => action?.Invoke(Window);
						}

					}

					_app.Startup += (sender, args) => _onWindowStartupExecuted = true;
				}

				_app.Startup += (sender, args) =>
				{
					reset.Set();
				};

				try
				{
					_app.Run(Window);
				}
				catch (Exception e)
				{
					_log.Fatal("Uncaught exception in Sigma UI has been thrown.", e);
					throw;
				}
			});

			_wpfThread = wpfThread.Thread;

			//Start the new thread with the given priority and set it to a STAThread (required for WPF windows)
			wpfThread.Thread.SetApartmentState(ApartmentState.STA);
			wpfThread.Thread.Priority = Priority;
			wpfThread.Thread.CurrentUICulture = _uiCultureInfo;
			wpfThread.Start();

			ColourManager.ForceUpdate();

			_log.Debug($"{nameof(WPFMonitor)} has been started. The window {_windowType.Name} should now be visible. ");
		}

		/// <inheritdoc />
		public override void SignalStop()
		{
			if (StopSigmaOnClose)
			{
				Sigma.SignalStop();
			}

			Dispose();
		}

		/// <summary>
		/// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
		/// </summary>
		public override void Dispose()
		{
			base.Dispose();
			Sigma.RemoveMonitor(this);
		}

		#endregion Lifecyclce

		#region WindowDispatch

		private void InvokeWindowDispatcherAsync<T>(Action<T> action, DispatcherPriority priority) where T : WPFWindow
		{
			Window.Dispatcher.InvokeAsync(() => action((T) Window), priority);
		}

		private void InvokeWindowDispatcher<T>(Action<T> action, DispatcherPriority priority) where T : WPFWindow
		{
			Window.Dispatcher.Invoke(() => action((T) Window), priority);
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow" />.
		/// All commands will be executed in the thread of the window!
		/// If the environment has note been prepared, the function will be executed
		/// in OnStartup function of the window.
		/// 
		/// 		The method will block until fully executed (if the window has already been started).
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="invokeWindowDispatcher">The action that executes the window dispatchment (either synchronous or asynchronous).</param>
		/// <param name="priority">The priority of the execution.</param>
		private void WindowDispatcher<T>(Action<T> action, Action<Action<T>, DispatcherPriority> invokeWindowDispatcher, DispatcherPriority priority = DispatcherPriority.Normal) where T : WPFWindow
		{
			if (typeof(T) != _windowType)
			{
				throw new ArgumentException($"Type mismatch between {typeof(T)} and {_windowType}");
			}

			if (Window == null)
			{
				_onWindowStartup.Add(obj => action((T) obj));
			}
			else
			{
				try
				{
					invokeWindowDispatcher(action, priority);
				}
				catch (TaskCanceledException e)
				{
					_log.Error("A window dispatcher method has been cancelled. This happens most likely when the window is closed, but a task still executing / newly added.", e);
					throw;
				}
			}
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow" />.
		/// All commands will be executed in the thread of the window! (i.e. blocking until finished, use <see cref="WindowDispatcherAsync{T}(Action{T},DispatcherPriority)"/> otherwise)
		/// If the environment has note been prepared, the function will be executed
		/// in OnStartup function of the window.
		/// 
		/// The method will block until fully executed (if the window has already been started).
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		public void WindowDispatcher<T>(Action<T> action, DispatcherPriority priority = DispatcherPriority.Normal) where T : WPFWindow
		{
			WindowDispatcher(action, InvokeWindowDispatcher, priority);
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow" />.
		/// All commands will be executed in the thread of the window! (i.e. blocking until finished, use <see cref="WindowDispatcherAsync{T}(Action{T},DispatcherPriority)"/> otherwise)
		/// If the environment has note been prepared, the function will be executed
		/// in OnStartup function of the window.
		/// 
		/// The method will block until fully executed (if the window has already been started).
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		public void WindowDispatcher(Action<SigmaWindow> action, DispatcherPriority priority = DispatcherPriority.Normal)
		{
			WindowDispatcher<SigmaWindow>(action, priority);
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow" />.
		/// All commands will be executed asynchronously! (i.e. use <see cref="WindowDispatcher{T}(Action{T},DispatcherPriority)"/> otherwise)
		/// If the environment has note been prepared, the function will be executed
		/// in OnStartup function of the window.
		/// 
		/// The method will return immediately.
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		public void WindowDispatcherAsync<T>(Action<T> action, DispatcherPriority priority = DispatcherPriority.Normal) where T : WPFWindow
		{
			WindowDispatcher(action, InvokeWindowDispatcherAsync, priority);
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow" />.
		/// All commands will be executed asynchronously! (i.e. use <see cref="WindowDispatcher{T}(Action{T},DispatcherPriority)"/> otherwise)
		/// If the environment has note been prepared, the function will be executed
		/// in OnStartup function of the window.
		/// 
		/// The method will return immediately.
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		public void WindowDispatcherAsync(Action<SigmaWindow> action, DispatcherPriority priority = DispatcherPriority.Normal)
		{
			WindowDispatcherAsync<SigmaWindow>(action, priority);
		}

		#endregion WindowDispatch

		#region Logging

		/// <summary>
		/// Assign the appender to the log. 
		/// </summary>
		protected virtual void AssignToLog()
		{
			((Hierarchy) LogManager.GetRepository()).Root.AddAppender(this);
		}

		/// <summary>Closes the appender and releases resources.</summary>
		/// <remarks>
		/// <para>
		/// Releases any resources allocated within the appender such as file handles,
		/// network connections, etc.
		/// </para>
		/// <para>
		/// It is a programming error to append to a closed appender.
		/// </para>
		/// </remarks>
		public void Close()
		{

		}

		/// <summary>Log the logging event to the root window (see <see cref="Window"/>). Requires an 
		/// <see cref="FilterDecision.Accept"/> from <see cref="LogFilter"/> (or unset <see cref="LogFilter"/>).</summary>
		/// <param name="loggingEvent">The event to log</param>
		/// <remarks>
		/// <para>
		/// This method is called to log a message into this appender.
		/// </para>
		/// </remarks>
		public void DoAppend(LoggingEvent loggingEvent)
		{
			if (LogFilter == null || LogFilter.Decide(loggingEvent) == FilterDecision.Accept)
			{
				Window.DoAppend(loggingEvent);
			}
		}

		/// <summary>Gets or sets the name of this appender.</summary>
		/// <value>The name of the appender.</value>
		/// <remarks>
		/// <para>The name uniquely identifies the appender.</para>
		/// </remarks>
		public string Name
		{
			get { return Title; }
			set { Title = value; }
		}

		#endregion Logging
	}
}