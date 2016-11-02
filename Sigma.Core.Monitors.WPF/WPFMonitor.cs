using System.Threading;
using Sigma.Core.Monitors.WPF.View;

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
		/// When the <see cref="Start"/> method is called, the thread should block
		/// until WPF is up and running.
		/// </summary>
		private ManualResetEvent waitForStart;

		/// <summary>
		/// The title of the window.
		/// </summary>
		private string title;

		/// <summary>
		/// Set or get the title of the window. 
		/// </summary>
		public string Title
		{
			get
			{
				return title;
			}
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
		/// The constructor for the WPF Monitor.
		/// </summary>
		/// <param name="title">The title of the new window.</param>
		public WPFMonitor(string title)
		{
			Title = title;

			waitForStart = new ManualResetEvent(false);
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

			wpfThread.SetApartmentState(ApartmentState.STA);
			wpfThread.Priority = Priority;
			wpfThread.Start();

			waitForStart.WaitOne();
			waitForStart.Reset();
		}
	}
}
