namespace Sigma.Core.Monitors
{
	public abstract class MonitorAdapter : IMonitor
	{
		public void Initialise()
		{

		}

		public abstract void Start();

		public virtual void Dispose()
		{
		}

		~MonitorAdapter()
		{
			Dispose();
		}
	}
}
