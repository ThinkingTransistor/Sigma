/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using MahApps.Metro.Controls;

namespace Sigma.Core.Monitors.WPF.View.Windows
{

	public abstract class WpfWindow : MetroWindow
	{
		/// <summary>
		/// The corresponding WPFMonitor
		/// </summary>
		protected WPFMonitor Monitor;

		/// <summary>
		/// The root application environment for all WPF interactions. 
		/// </summary>
		public App @App { get; }

		/// <summary>
		/// The constructor for the <see cref="WpfWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="System.Windows.Application"/> environment.</param>
		/// <param name="title">The <see cref="System.Windows.Window.Title"/> of the window.</param>
		public WpfWindow(WPFMonitor monitor, App app, string title) : base()
		{
			CheckArgs(monitor, app);

			Monitor = monitor;
			App = app;
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
			if (monitor == null) throw new ArgumentNullException(nameof(monitor));
			if (app == null) throw new ArgumentNullException(nameof(app));
		}

		/// <summary>
		/// In this function components should be initialised that
		/// don't depend on constructor arguments. This function
		/// will be invoked as the last operation
		/// of the <see cref="WpfWindow"/>'s constructor. 
		/// </summary>
		protected abstract void InitialiseComponents();
	}
}
