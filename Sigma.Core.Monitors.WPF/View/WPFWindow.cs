/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using MahApps.Metro.Controls;

namespace Sigma.Core.Monitors.WPF.View
{

	public class WPFWindow : MetroWindow
	{
		/// <summary>
		/// The corresponding WPFMonitor
		/// </summary>
		private WPFMonitor monitor;

		/// <summary>
		/// The app-environment. 
		/// </summary>
		private App app;

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
		/// The constructor for the WPF window.
		/// </summary>
		/// <param name="title">The title of the window.</param>
		public WPFWindow(WPFMonitor monitor, App app, string title) : base()
		{
			this.monitor = monitor;
			this.app = app;
			Title = title;

			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			//This can only be set in the constructor or onstartup
			BorderThickness = new Thickness(0);
			GlowBrush = Brushes.Black;

			//GlowBrush = app.FindResource("AccentColorBrush") as Brush;
		}
	}
}
