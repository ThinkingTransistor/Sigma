using System;

namespace Sigma.Core.Monitors
{
	public interface IMonitor : IDisposable
	{
		/// <summary>
		/// This function will be called before the first use of the monitor.
		/// </summary>
		void Initialise();
	}
}
