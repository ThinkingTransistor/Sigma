namespace Sigma.Core.Monitors
{
	public abstract class MonitorAdapter : IMonitor
	{
		public abstract void Initialise();

		public virtual void Dispose()
		{

		}

		~MonitorAdapter()
		{
			Dispose();
		}
	}
}
