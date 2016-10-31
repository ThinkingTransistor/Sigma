using System.Threading;
using System.Windows;

namespace Sigma.Core.Monitors.WPF
{
	internal class WPFController : Window
	{
		private WPFController()
		{

		}

		/// <summary>
		/// Start the WPF window in the same thread.
		/// </summary>
		internal static void Start()
		{
			new Application().Run(new WPFController());
		}

		/// <summary>
		/// Start the WPF window asynchronously. 
		/// </summary>
		internal static void StartInNewThread(ThreadPriority priority = ThreadPriority.Normal)
		{
			Thread wpfThread = new Thread(() => Start());

			wpfThread.SetApartmentState(ApartmentState.STA);

			wpfThread.Priority = priority;

			wpfThread.Start();
		}
	}
}
