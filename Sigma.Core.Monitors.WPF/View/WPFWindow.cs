/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using Sigma.Core.Monitors.WPF.Control;

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
