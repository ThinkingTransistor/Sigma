using System.Threading;
using NUnit.Framework;
// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.Tests
{
	public class WPFMonitorTest
	{
		private static SigmaEnvironment ClearAndCreate(string identifier)
		{
			SigmaEnvironment.Clear();

			return SigmaEnvironment.Create(identifier);
		}

		[TestCase]
		public void TestWPFMonitorCreation()
		{
			SigmaEnvironment sigma = ClearAndCreate("Test");

			WPFMonitor monitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));
			monitor.Priority = ThreadPriority.Lowest;

			Assert.AreSame(sigma, monitor.Sigma);
			Assert.AreEqual(monitor.Priority, ThreadPriority.Lowest);
			Assert.AreEqual(monitor.Title, "Sigma GUI Demo");
			Assert.AreNotEqual(monitor.Title, "Sigma GUI Demo2");
		}
	}
}
