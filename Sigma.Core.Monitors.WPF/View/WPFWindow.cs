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
using Dragablz.Dockablz;
using MahApps.Metro.Controls;
using Sigma.Core.Monitors.WPF.Control.Tabs;
using Sigma.Core.Monitors.WPF.Model.UI;
using Sigma.Core.Monitors.WPF.View.Tabs;

namespace Sigma.Core.Monitors.WPF.View
{

	public class WPFWindow : MetroWindow
	{
		#region DependencyProperties

		public static readonly DependencyProperty DefaultGridSizeProperty = DependencyProperty.Register("DefaultGridSize", typeof(GridSize), typeof(WPFWindow), new UIPropertyMetadata(new GridSize(3, 4)));

		#endregion DependencyProperties

		#region Properties

		/// <summary>
		/// The DefaultGridSize for each newly created <see cref="Tab"/>.
		/// The default <see cref="DefaultGridSize"/> is 3, 4.
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

		/// <summary>
		/// The <see cref="TabControl"/> for the tabs. It allows to access each <see cref="TabUI"/>
		/// and therefore, the <see cref="TabItem"/>.
		/// </summary>
		public TabControlUI TabControl { get; set; }

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		public WPFWindow(WPFMonitor monitor, App app, string title) : this(monitor, app, title, true) { }

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		/// <param name="addTabs">Decides whether the saved <see cref="WPFMonitor.Tabs"/> should be added or not. </param>
		internal WPFWindow(WPFMonitor monitor, App app, string title, bool addTabs) : base()
		{
			CheckArgs(monitor, app);

			this.monitor = monitor;
			this.app = app;
			Title = title;

			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			SetBorderBehaviour(app);

			AddResources();

			TabControl = CreateTabControl();

			if (addTabs)
			{
				//HACK: not Thread safe, if user is stupid and adds tabs 
				//to the registry after start
				AddTabs(TabControl, monitor.Tabs);
			}

			Content = (Layout) TabControl;
		}

		/// <summary>
		/// Check whether the args are correct.
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
		/// Define how the border of the application behaves.
		/// </summary>
		/// <param name="app">The app environemnt. </param>
		private void SetBorderBehaviour(App app)
		{
			//This can only be set in the constructor or onstartup
			Brush accentColorBrush = app.FindResource("AccentColorBrush") as Brush;

			BorderThickness = new Thickness(1);
			BorderBrush = accentColorBrush;
			GlowBrush = accentColorBrush;

			//Disable that the titlebar will get grey if not focused. 
			//And any other changes that may occur when the window is not focused.
			NonActiveWindowTitleBrush = accentColorBrush;
			NonActiveBorderBrush = BorderBrush;
			NonActiveGlowBrush = GlowBrush;
		}

		/// <summary>
		/// This function adds all required resources. 
		/// </summary>
		private void AddResources()
		{

		}

		/// <summary>
		/// THis function creates the <see cref="TabControlUI"/>.
		/// </summary>
		/// <returns>The newly created <see cref="TabControlUI"/>.</returns>
		private TabControlUI CreateTabControl()
		{
			return new TabControlUI(monitor, app, Title);
		}

		/// <summary>
		/// Adds the tabs to the given <see cref="TabControlUI"/>.
		/// </summary>
		/// <param name="tabControl">The <see cref="TabControlUI"/>, where the <see cref="TabItem"/>s will be added to.</param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		private void AddTabs(TabControlUI tabControl, List<string> names)
		{
			for (int i = 0; i < names.Count; i++)
			{
				tabControl.AddTab(new TabUI(names[i]));
			}
		}
	}
}
