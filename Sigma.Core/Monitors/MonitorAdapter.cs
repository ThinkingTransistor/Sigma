/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

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
