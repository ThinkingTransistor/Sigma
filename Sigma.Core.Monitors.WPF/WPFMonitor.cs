/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Threading;
using Sigma.Core.Monitors.WPF.Control.Themes;
using Sigma.Core.Monitors.WPF.View;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF
{
	/// <summary>
	/// This <see cref="IMonitor"/> is the default visualisation monitor shipped with the big Sigma.
	/// The <see cref="WPFMonitor"/> is designed to run on Windows.
	/// </summary>
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
		private App app;

		/// <summary>
		/// The active WPF window that will be maintained. 
		/// </summary>
		private WPFWindow window;

		/// <summary>
		/// This property returns the current window. 
		/// <see cref="Window"/> is <see langword="null"/> until <see cref="SigmaEnvironment.Prepare"/> has been called.
		/// </summary>
		internal WPFWindow Window
		{
			get { return window; }
		}

		/// <summary>
		/// When the <see cref="Start"/> method is called, the thread should block
		/// until WPF is up and running.
		/// </summary>
		private ManualResetEvent waitForStart;

		/// <summary>
		/// The title of the window.
		/// </summary>
		private string title;

		/// <summary>
		/// Property for the title of the window. 
		/// </summary>
		public string Title
		{
			get { return title; }
			set
			{
				title = value;

				if (window != null)
				{
					window.Title = title;
				}
			}
		}

		private Type windowType;

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
		public IColorManager @ColorManager
		{
			get;
			private set;
		}

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
				throw new ArgumentException($"Type {window} does not extend from {typeof(WPFWindow)}");
			}

			Title = title;
			windowType = window;

			ColorManager = new ColorManager(MaterialDesignSwatches.BLUE, MaterialDesignSwatches.AMBER);

			waitForStart = new ManualResetEvent(false);
		}

		public override void Initialise()
		{
			//Tabs = new TabRegistry(Sigma.Registry);
			Tabs = new List<string>();
		}

		public override void Start()
		{
			Thread wpfThread = new Thread(() =>
			{
				app = new App(this);
				ColorManager.App = app;

				window = (WPFWindow) Activator.CreateInstance(windowType, this, app, title);

				app.Startup += (sender, args) => waitForStart.Set();

				app.Run(window);
			});

			//Start the new thread with the given priority and set it to a STAThread (required for WPF windows)
			wpfThread.SetApartmentState(ApartmentState.STA);
			wpfThread.Priority = Priority;
			wpfThread.Start();

			//Wait until the thread has finished execution
			waitForStart.WaitOne();
			waitForStart.Reset();
		}

		/// <summary>
		/// This method allows to access the <see cref="WPFWindow"/>. 
		/// All commands will be executed in the thread of the window!
		/// </summary>
		/// <param name="action">The action that should be executed from the <see cref="WPFWindow"/>.</param>
		/// <param name="priority">The priority of the execution.</param>
		/// <param name="onFinished">The action that should be called after the action has been finished. This action will be called from the caller thread.</param>
		public void WindowDispatcher(Action<WPFWindow> action, DispatcherPriority priority = DispatcherPriority.Normal, Action onFinished = null)
		{
			window.Dispatcher.Invoke(() =>
			{
				action(window);
			});

			if (onFinished != null)
			{
				throw new NotImplementedException($"{nameof(onFinished)} action not yet implemented... Sorry");
			}
		}
	}
}
