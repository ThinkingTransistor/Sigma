using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace Sigma.Core.Monitors.WPF.View
{
	public class WPFWindow : Window
	{
		/// <summary>
		/// The constructor for the WPF window.
		/// </summary>
		/// <param name="title">The title of the window.</param>
		public WPFWindow(string title)
		{
			Title = title;
		}
	}
}
