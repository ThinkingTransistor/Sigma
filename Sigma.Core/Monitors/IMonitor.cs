namespace Sigma.Core.Monitors
{
	public interface IMonitor
	{
		/// <summary>
		/// This function will be called before the first use of the monitor.
		/// </summary>
		void Initialise();

		/// <summary>
		/// In this funciton, all accessed ressources should be released.
		/// Additionally, call this function in the Destructor.
		/// </summary>
		void Destroy();
	}
}
