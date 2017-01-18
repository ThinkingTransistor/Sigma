/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Threading;
using System.Windows;
using System.Windows.Threading;

namespace Sigma.Core.Monitors.WPF.Utils
{
	public static class WindowUtils
	{
		/// <summary>
		/// This methods dispatches a given command in the thread of the window. 
		/// If its already the correct thread, it will be executed in the current one. 
		/// </summary>
		/// <param name="window">The window the action will be performed on.</param>
		/// <param name="command">The command that will be executed.</param>
		public static void DispatchCommand(this Window window, Action command)
		{
			if (command == null)
			{
				throw new ArgumentNullException(nameof(command));
			}

			if (Dispatcher.CurrentDispatcher.Thread == Thread.CurrentThread)
			{
				command();
			}
			else
			{
				window.Dispatcher.Invoke(command);
			}
		}
	}
}