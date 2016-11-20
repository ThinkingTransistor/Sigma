/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.WPF.Control.Themes;
using Sigma.Core.Monitors.WPF.View.Windows;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Threading;

namespace Sigma.Core.Monitors.WPF
{
	/// <summary>
	/// This <see cref="IMonitor"/> is the default visualisation monitor shipped with the big Sigma.
	/// The <see cref="WPFMonitor"/> is designed to run on Windows.
	/// </summary>
	// ReSharper disable once InconsistentNaming
	public class WPFMonitor : MonitorAdapter
	{
		/// <summary>
		/// The Priority with which the thread will be started.
		/// The default is <see cref="ThreadPriority.Normal"/>.
		/// </summary>
		public ThreadPriority Priority { get; set; } = ThreadPriority.Normal;

		/// <summary>
		/// The root application for all WPF interactions. 
		/// </summary>
		private App _app;

		/// <summary>
		/// This property returns the current window. 
		/// <see cref="Window"/> is <see langword="null"/> until <see cref="SigmaEnvironment.Prepare"/> has been called.
		/// </summary>
		public WPFWindow Window { get; private set; }

		/// <summary>
		/// When the <see cref="Start"/> method is called, the thread should block
		/// until WPF is up and running.
		/// </summary>
		private readonly ManualResetEvent _waitForStart;

		/// <summary>
		/// The title of the window.
		/// </summary>
		private string _title;

		/// <summary>
		/// Property for the title of the window. 
		/// </summary>
		public string Title
		{
			get { return _title; }
			set
			{
				_title = value;

				if (Window != null)
				{
					Window.Title = _title;
				}
			}
		}

		/// <summary>
		/// The type of the window that will be created.
		/// A type is passed in order to prevent generics.
		/// (Mostly that user does not have to care about it)
		/// </summary>
		private readonly Type _windowType;

		//HACK: decide what Tabs is
		/// <summary>
		/// The list of tabs that are available. These have to be set <b>before</b> <see cref="SigmaEnvironment.Prepare"/>.
		/// </summary>
		public List<string> Tabs { get; private set; }

		//HACK: decide what Tabs is
		public void AddTabs(params string[] tabs)
		{
			foreach (string tab in tabs)
			{
				Tabs.Add(tab);
			}
		}

		/// <summary>
		/// The <see cref="IColorManager"/> to control the look and feel of the application. 
		/// </summary>
		public IColorManager ColorManager
		{
			get;
		}

		/// <summary>
		/// Actions assigned to this list will listen
		/// to the onStart-event of the <see cref="_app"/>.
		/// </summary>
		private List<Action<object>> _onWindowStartup;

		/// <summary>
		/// The constructor for the WPF Monitor that relies on <see cref="SigmaWindow"/>.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		public WPFMonitor(string title) : this(title, typeof(SigmaWindow)) { }

		/// <summary>
		/// The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		/// <param name="window">The type of the <see cref="WPFWindow"/> that will be displayed. This window requires a constructor
		/// whit the same arguments as <see cref="WPFWindow"/>.</param>
		public WPFMonitor(string title, Type window)
		{
			if (!window.IsSubclassOf(typeof(WPFWindow)))
			{
				throw new ArgumentException($"Type {window} does not extend from {typeof(WPFWindow)}!");
			}

			Title = title;
			_windowType = window;

			ColorManager = new ColorManager(MaterialDesignSwatches.BLUE, MaterialDesignSwatches.AMBER);

			_waitForStart = new ManualResetEvent(false);
		}

		public override void Initialise()
		{
			base.Initialise();
			//Tabs = new TabRegistry(Sigma.Registry);
			Tabs = new List<string>();
		}

		public override void Start()
		{
			Thread wpfThread = new Thread(() =>
			{
				_app = new App(this);
				ColorManager.App = _app;

				Window = (WPFWindow) Activator.CreateInstance(_windowType, this, _app, _title);
				ColorManager.Window = Window;

				if (_onWindowStartup != null)
				{
					foreach (var action in _onWindowStartup)
					{
						_app.Startup += (sender, args) => action?.Invoke(Window);
					}
				}

				_app.Startup += (sender, args) =>
				{
					_waitForStart.Set();
				};
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
		/// This method allows to access the <see cref="WPFWindow"/>. 
		/// All commands will be executed in the thread of the window!
		/// If the environment has note been prepared, the function will be executed 
		/// in OnStartup function of the window. 
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow"/>.</param>
		/// <param name="priority">The priority of the execution.</param>
		/// <param name="onFinished">The action that should be called after the action has been finished. This action will be called from the caller thread.</param>
		/// <exception cref="NotImplementedException">Currently, <paramref name="onFinished"/> is not yet implemented.</exception>
		public void WindowDispatcher<T>(Action<T> action, DispatcherPriority priority = DispatcherPriority.Normal, Action onFinished = null) where T : WPFWindow
		{
			if (typeof(T) != _windowType) throw new ArgumentException($"Type mismatch between {typeof(T)} and {_windowType}");

			if (Window == null)
			{
				if (_onWindowStartup == null)
				{
					_onWindowStartup = new List<Action<object>>();
				}

				_onWindowStartup.Add((obj) => action((T) obj));
			}
			else
			{
				Window.Dispatcher.Invoke(() => action((T) Window), priority);
			}

			if (onFinished != null) throw new NotImplementedException($"{nameof(onFinished)} action not yet implemented... Sorry");
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow"/>. 
		/// All commands will be executed in the thread of the window!
		/// If the environment has note been prepared, the function will be executed 
		/// in OnStartup function of the window. 
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow"/>.</param>
		/// <param name="priority">The priority of the execution.</param>
		/// <param name="onFinished">The action that should be called after the action has been finished. This action will be called from the caller thread.</param>
		/// <exception cref="NotImplementedException">Currently, <paramref name="onFinished"/> is not yet implemented.</exception>
		public void WindowDispatcher(Action<SigmaWindow> action, DispatcherPriority priority = DispatcherPriority.Normal, Action onFinished = null)
		{
			WindowDispatcher<SigmaWindow>(action, priority, onFinished);
		}
	}
}
