/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors
{
	public interface IMonitor : IDisposable
	{
		/// <summary>
		/// This function will be called before the start of the monitor.
		/// </summary>
		void Initialise();

		/// <summary>
		/// Start this <see cref="IMonitor"/>.
		/// If the <see cref="IMonitor"/> runs in a new <see cref="System.Threading.Thread"/>, this function should block until the monitor is completely up and running. 
		/// </summary>
		void Start();

		/// <summary>
		/// Signal this <see cref="IMonitor"/> to stop. This method should call dispose.
		/// This could for example stop the Sigma learning process (if mandatory), or clear up registry entries, resources ...
		/// </summary>
		void SignalStop();

		/// <summary>
		/// The sigma environment associated with this monitor.
		/// </summary>
		SigmaEnvironment Sigma { get; set; }

		IRegistry Registry { get; set; }
	}
}
