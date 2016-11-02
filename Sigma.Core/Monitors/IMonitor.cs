/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

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
