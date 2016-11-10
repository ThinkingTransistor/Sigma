/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using MahApps.Metro.Controls;
using System;


namespace Sigma.Core.Monitors.WPF.View.Windows
{

	public abstract class WPFWindow : MetroWindow
	{
		/// <summary>
		/// The corresponding WPFMonitor
		/// </summary>
		protected WPFMonitor monitor;

		/// <summary>
		/// The app-environment. 
		/// </summary>
		protected App app;

		/// <summary>
		/// The root application environment for all WPF interactions. 
		/// </summary>
		public App @App
		{
			get
			{
				return app;
			}
		}

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		public WPFWindow(WPFMonitor monitor, App app, string title) : base()
		{
			CheckArgs(monitor, app);

			this.monitor = monitor;
			this.app = app;
			Title = title;

			InitialiseComponents();
		}


		/// <summary>
		/// Check whether the arguments are correct.
		/// Returns or throws Exception. 
		/// </summary>
		/// <param name="monitor">The <see cref="WPFMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		private static void CheckArgs(WPFMonitor monitor, App app)
		{
			if (monitor == null)
			{
				throw new ArgumentNullException("Monitor may not be null!");
			}

			if (app == null)
			{
				throw new ArgumentNullException("App may not be null");
			}
		}

		/// <summary>
		/// In this function components should be initialised that
		/// don't depend on constructor arguments. This function
		/// will be invoked as the last operation
		/// of the <see cref="WPFWindow"/>'s constructor. 
		/// </summary>
		protected abstract void InitialiseComponents();
	}
}
