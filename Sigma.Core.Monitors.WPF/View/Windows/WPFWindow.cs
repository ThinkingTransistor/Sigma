/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using MahApps.Metro.Controls;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	// ReSharper disable once InconsistentNaming
	public abstract class WPFWindow : MetroWindow
	{
		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		protected WPFWindow(WPFMonitor monitor, Application app, string title)
		{
			if (monitor == null)
			{
				throw new ArgumentNullException(nameof(monitor));
			}
			if (app == null)
			{
				throw new ArgumentNullException(nameof(app));
			}

			Monitor = monitor;
			App = app;
			Title = title;

			InitialiseComponents();
		}

		/// <summary>
		///     The corresponding WPFMonitor
		/// </summary>
		public WPFMonitor Monitor { get; protected set; }

		/// <summary>
		///     The root application environment for all WPF interactions.
		/// </summary>
		public Application App { get; }


		/// <summary>
		///     In this function components should be initialised that
		///     don't depend on constructor arguments. This function
		///     will be invoked as the last operation
		///     of the <see cref="WPFWindow" />'s constructor.
		/// </summary>
		protected abstract void InitialiseComponents();

		/// <summary>
		///     This method handles all unhandled dispatcher exceptions (see <see cref="UnhandledExceptionEventArgs" />).
		/// </summary>
		/// <param name="sender">The sender of the exception.</param>
		/// <param name="e">The information of the exception.</param>
		public abstract void HandleUnhandledException(object sender, UnhandledExceptionEventArgs e);
	}
}