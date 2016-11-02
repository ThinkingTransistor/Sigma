/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

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
