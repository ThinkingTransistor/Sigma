/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Dragablz;
using Dragablz.Dockablz;
using MahApps.Metro.Controls;
using Sigma.Core.Monitors.WPF.Model.UI;
using Sigma.Core.Monitors.WPF.View.Tabs;

namespace Sigma.Core.Monitors.WPF.View
{

	public class WPFWindow : MetroWindow
	{
		#region DependencyProperties

		public static readonly DependencyProperty DefaultGridSizeProperty = DependencyProperty.Register("DefaultGridSize", typeof(GridSize), typeof(WPFWindow), new UIPropertyMetadata(new GridSize()));

		#endregion DependencyProperties

		#region Properties

		/// <summary>
		/// The DefaultGridSize for each newly created <see cref="Tab"/>.
		/// </summary>
		public GridSize DefaultGridSize
		{
			get { return (GridSize) GetValue(DefaultGridSizeProperty); }
			set { SetValue(DefaultGridSizeProperty, value); }
		}

		#endregion Properties

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

		private TabControlUI tabControl; 

		//private TabablzControl tabControl;

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
			Brush accentColorBrush = app.FindResource("AccentColorBrush") as Brush;

			BorderThickness = new Thickness(1);
			BorderBrush = accentColorBrush;
			GlowBrush = accentColorBrush;

			//HACK: not Thread safe, if user is stupid and adds tabs 
			//to the registry after start
			tabControl = AddTabs(monitor.Tabs);

			Content = (Layout) tabControl;
		}

		private TabControlUI AddTabs(List<string> names)
		{
			TabControlUI tabControl = new TabControlUI();

			for (int i = 0; i < names.Count; i++)
			{
				tabControl.AddTab(new TabUI(names[i]));
			}
			
			return tabControl;
		}
	}
}
