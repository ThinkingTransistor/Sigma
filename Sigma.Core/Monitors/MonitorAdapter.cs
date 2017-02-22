/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;

namespace Sigma.Core.Monitors
{
	public abstract class MonitorAdapter : IMonitor
	{
		public SigmaEnvironment Sigma { get; set; }
		public IRegistry Registry { get; set; }

		public virtual void Initialise() { }

		public abstract void Start();

		public virtual void Dispose() { }

		public virtual void SignalStop() { }
	}
}
