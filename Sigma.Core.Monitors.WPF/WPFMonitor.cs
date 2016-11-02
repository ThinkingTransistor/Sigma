/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Monitors.WPF
{
	/// <summary>
	/// The <see cref="WPFMonitor"/> is designed to run on Windows.
	/// </summary>
	public class WPFMonitor : MonitorAdapter
	{
		public override void Initialise()
		{
			WPFController.StartInNewThread();
		}
	}
}
