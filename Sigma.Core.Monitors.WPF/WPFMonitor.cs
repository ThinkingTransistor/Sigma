/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Threading;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.View.Themes;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF
{
	/// <summary>
	///     This <see cref="IMonitor" /> is the default visualisation monitor shipped with the big Sigma.
	///     The <see cref="WPFMonitor" /> is designed to run on Windows.
	/// </summary>
	// ReSharper disable once InconsistentNaming
	public class WPFMonitor : MonitorAdapter
	{
		/// <summary>
		///     Actions assigned to this list will listen
		///     to the onStart-event of the <see cref="_app" />.
		/// </summary>
		private readonly List<Action<object>> _onWindowStartup = new List<Action<object>>();

		/// <summary>
		///     When the <see cref="Start" /> method is called, the thread should block
		///     until WPF is up and running.
		/// </summary>
		private readonly ManualResetEvent _waitForStart;

		/// <summary>
		///     The type of the window that will be created.
		///     A type is passed in order to prevent generics.
		///     (Mostly that user does not have to care about it)
		/// </summary>
		private readonly Type _windowType;

		/// <summary>
		///     The root application for all WPF interactions.
		/// </summary>
		private App _app;

		/// <summary>
		///     This <see cref="bool" /> will be set to true,
		///     as soon as all <see cref="_onWindowStartup" />
		///     actions have been added.
		/// </summary>
		/// TODO: HACK: 
#pragma warning disable 414
		private bool _onWindowStartupAdded;
#pragma warning restore 414

		/// <summary>
		///     This <see cref="bool" /> will be set to true,
		///     as soon as all <see cref="_onWindowStartup" />
		///     actions have been executed.
		/// </summary>
		private bool _onWindowStartupExecuted;

		/// <summary>
		///     The title of the window.
		/// </summary>
		private string _title;

		/// <summary>
		///     The constructor for the WPF Monitor that relies on <see cref="SigmaWindow" />.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		public WPFMonitor(string title) : this(title, typeof(SigmaWindow))
		{
		}

		/// <summary>
		///     The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="window">
		///     The type of the <see cref="WPFWindow" /> that will be displayed. This window requires a constructor
		///     whit the same arguments as <see cref="WPFWindow" />.
		/// </param>
		public WPFMonitor(string title, Type window)
		{
			if (!window.IsSubclassOf(typeof(WPFWindow)))
				throw new ArgumentException($"Type {window} does not extend from {typeof(WPFWindow)}!");

			Title = title;
			_windowType = window;

			// this is the one and only appropriate colour configuration - fight me
			ColourManager = new ColourManager(MaterialDesignValues.BlueGrey, MaterialDesignValues.Amber);
			ColourManager.Dark = true;

			_waitForStart = new ManualResetEvent(false);
		}

		/// <summary>
		///     The Priority with which the thread will be started.
		///     The default is <see cref="ThreadPriority.Normal" />.
		/// </summary>
		public ThreadPriority Priority { get; set; } = ThreadPriority.Normal;

		/// <summary>
		///     This property returns the current window.
		///     <see cref="Window" /> is <see langword="null" /> until <see cref="SigmaEnvironment.Prepare" /> has been called.
		/// </summary>
		public WPFWindow Window { get; private set; }

		/// <summary>
		///     Property for the title of the window.
		/// </summary>
		public string Title
		{
			get { return _title; }
			set
			{
				_title = value;

				if (Window != null)
					Window.Title = _title;
			}
		}

		//HACK: decide what Tabs is
		/// <summary>
		///     The list of tabs that are available. These have to be set <b>before</b> <see cref="SigmaEnvironment.Prepare" />.
		/// </summary>
		public List<string> Tabs { get; private set; }

		//HACK: 
		//TODO: documentation

		internal Dictionary<string, StatusBarLegendInfo> Legends { get; private set; }

		/// <summary>
		///     The <see cref="IColourManager" /> to control the look and feel of the application.
		/// </summary>
		public IColourManager ColourManager { get; }

		//HACK: decide what Tabs is
		public void AddTabs(params string[] tabs)
		{
			Tabs.AddRange(tabs);
		}

		public StatusBarLegendInfo AddLegend(StatusBarLegendInfo legend)
		{
			AddLegends(legend);

			return GetLegendInfo(legend.Name);
		}

		public void AddLegends(params StatusBarLegendInfo[] legends)
		{
			lock (_onWindowStartup)
			{
				if (_onWindowStartupExecuted)
				{
					throw new NotImplementedException("Window has already been started - StatusBar update not supported (yet).");
				}

				AddLegends(Legends, legends);
			}
		}

		public StatusBarLegendInfo GetLegendInfo(string name)
		{
			return Legends[name];
		}

		public IEnumerable<StatusBarLegendInfo> GetLegends()
		{
			return Legends.Values;
		}

		private static void AddLegends(IDictionary<string, StatusBarLegendInfo> legends, StatusBarLegendInfo[] legendInfo)
		{
			foreach (StatusBarLegendInfo statusBarLegendInfo in legendInfo)
			{
				legends.Add(statusBarLegendInfo.Name, statusBarLegendInfo);
			}
		}

		public override void Initialise()
		{
			base.Initialise();

			Tabs = new List<string>();
			Legends = new Dictionary<string, StatusBarLegendInfo>();
		}

		public override void Start()
		{
			Thread wpfThread = new Thread(() =>
			{
				_app = new App(this);
				ColourManager.App = _app;

				Window = (WPFWindow) Activator.CreateInstance(_windowType, this, _app, _title);

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

					_onWindowStartupAdded = true;
				}

				_app.Startup += (sender, args) => { _waitForStart.Set(); };
				_app.Run(Window);
			});

			//Start the new thread with the given priority and set it to a STAThread (required for WPF windows)
			wpfThread.SetApartmentState(ApartmentState.STA);
			wpfThread.Priority = Priority;
			wpfThread.Start();

			//Wait until the thread has finished execution
			_waitForStart.WaitOne();
			_waitForStart.Reset();
		}

		/// <summary>
		///     This method allows to access the <see cref="WPFWindow" />.
		///     All commands will be executed in the thread of the window!
		///     If the environment has note been prepared, the function will be executed
		///     in OnStartup function of the window.
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		/// <param name="onFinished">
		///     The action that should be called after the action has been finished. This action will be
		///     called from the caller thread.
		/// </param>
		/// <exception cref="NotImplementedException">Currently, <paramref name="onFinished" /> is not yet implemented.</exception>
		public void WindowDispatcher<T>(Action<T> action, DispatcherPriority priority = DispatcherPriority.Normal,
			Action onFinished = null) where T : WPFWindow
		{
			if (typeof(T) != _windowType) throw new ArgumentException($"Type mismatch between {typeof(T)} and {_windowType}");

			if (Window == null)
			{
				_onWindowStartup.Add(obj => action((T) obj));
			}
			else
			{
				Window.Dispatcher.Invoke(() => action((T) Window), priority);
			}

			if (onFinished != null)
			{
				throw new NotImplementedException($"{nameof(onFinished)} not yet implemented... Sorry");
			}
		}

		/// <summary>
		///     This method allows to access the <see cref="WPFWindow" />.
		///     All commands will be executed in the thread of the window!
		///     If the environment has note been prepared, the function will be executed
		///     in OnStartup function of the window.
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow" />.</param>
		/// <param name="priority">The priority of the execution.</param>
		/// <param name="onFinished">
		///     The action that should be called after the action has been finished. This action will be
		///     called from the caller thread.
		/// </param>
		/// <exception cref="NotImplementedException">Currently, <paramref name="onFinished" /> is not implemented.</exception>
		public void WindowDispatcher(Action<SigmaWindow> action, DispatcherPriority priority = DispatcherPriority.Normal,
			Action onFinished = null)
		{
			WindowDispatcher<SigmaWindow>(action, priority, onFinished);
		}
	}
}