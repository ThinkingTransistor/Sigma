/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using log4net.Appender;
using log4net.Core;
using MahApps.Metro.Controls;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	// ReSharper disable once InconsistentNaming
	public abstract class WPFWindow : MetroWindow, IAppender
	{
		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		/// <param name="title">The <see cref="Window.Title" /> of the window.</param>
		protected WPFWindow(WPFMonitor monitor, Application app, string title) : this(monitor, app)
		{
			Title = title;
		}

		/// <summary>
		///     The constructor for the <see cref="WPFWindow" />.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor" />.</param>
		/// <param name="app">The <see cref="Application" /> environment.</param>
		protected WPFWindow(WPFMonitor monitor, Application app)
		{
			if (monitor == null) throw new ArgumentNullException(nameof(monitor));
			if (app == null) throw new ArgumentNullException(nameof(app));

			Monitor = monitor;
			App = app;

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
		/// This boolean determines whether the UI should be fully shown (<see cref="IsInitializing"/> = <c>false</c>)
		/// or a loading indicator / nothing (depending on the window) is visible. 
		/// </summary>
		public abstract bool IsInitializing { get; set; }

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

		/// <summary>Log the logging event in Appender specific way. Do not assign to the log, the monitor will pass logs to its set window. </summary>
		/// <param name="loggingEvent">The event to log</param>
		/// <remarks>
		/// <para>
		/// This method is called to log a message into this appender.
		/// </para>
		/// </remarks>
		public abstract void DoAppend(LoggingEvent loggingEvent);
	}
}