/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Threading;
using Sigma.Core.Monitors.WPF.Control;
using Sigma.Core.Monitors.WPF.View;
using Sigma.Core.Utils;

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
		public WPFWindow Window
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

		/// <summary>
		/// The <see cref="TabControl"/> that allows to access all <see cref="Tab"/>s.
		/// It is <see langword="null"/> until the corresponding <see cref="IMonitor"/> has been added to the <see cref="SigmaEnvironment"/>.
		/// </summary>
		public TabRegistry Tabs { get; private set; }

		/// <summary>
		/// The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		public WPFMonitor(string title)
		{
			Title = title;

			waitForStart = new ManualResetEvent(false);
		}

		public override void Initialise()
		{
			Tabs = new TabRegistry(Sigma.Registry);
		}

		public override void Start()
		{
			Thread wpfThread = new Thread(() =>
			{
				window = new WPFWindow(title);
				app = new App();
				app.Run(window);

				waitForStart.Set();
			});

			//Start the new thread with the given priority and set it to a STAThread (required for some WPF windows)
			wpfThread.SetApartmentState(ApartmentState.STA);
			wpfThread.Priority = Priority;
			wpfThread.Start();

			//Wait until the thread has finished execution
			waitForStart.WaitOne();
			waitForStart.Reset();
		}
	}
}
