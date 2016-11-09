using System.Collections.Generic;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Dragablz.Dockablz;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Control.Tabs;
using Sigma.Core.Monitors.WPF.Control.TitleBar;
using Sigma.Core.Monitors.WPF.Model.UI;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.View.Tabs;
using Sigma.Core.Monitors.WPF.View.TitleBar;

namespace Sigma.Core.Monitors.WPF.View.Windows
{
	public class SigmaWindow : WPFWindow
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
		/// The <see cref="TitleBarControl"/> for the dropdwons in the title.
		/// With this property you can access every button of the dropdown.
		/// </summary>
		public TitleBarControl TitleBar { get; private set; }

		/// <summary>
		/// The <see cref="TabControl"/> for the tabs. It allows to access each <see cref="TabUI"/>
		/// and therefore, the <see cref="TabItem"/>.
		/// </summary>
		public TabControlUI<SigmaWindow> TabControl { get; set; }

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		public SigmaWindow(WPFMonitor monitor, App app, string title) : this(monitor, app, title, true)
		{

		}

		/// <summary>
		/// The constructor for the <see cref="WPFWindow"/>.
		/// </summary>
		/// <param name="monitor">The root <see cref="IMonitor"/>.</param>
		/// <param name="app">The <see cref="Application"/> environment.</param>
		/// <param name="title">The <see cref="Window.Title"/> of the window.</param>
		/// <param name="addTabs">Decides whether the saved <see cref="WPFMonitor.Tabs"/> should be added or not. </param>
		protected SigmaWindow(WPFMonitor monitor, App app, string title, bool addTabs) : base(monitor, app, title)
		{
			FontFamily = UIColours.FontFamily;

			TitleAlignment = HorizontalAlignment.Center;

			TitleBar = CreateTitleBar();
			LeftWindowCommands = TitleBar;

			//Button button = TitleBar[0].Elements[0] as Button;
			//PopupBox box = TitleBar[0].Content as PopupBox;

			//Debug.WriteLine(box.FocusVisualStyle == null);

			//Debug.WriteLine($"Height: {button.ActualHeight}, Width: {button.ActualWidth}");
			//var triggers = box.Toggle

			//foreach (var trigger in triggers)
			//{
			//	Debug.WriteLine(trigger);
			//}


			TabControl = CreateTabControl();

			if (addTabs)
			{
				//HACK: not Thread safe, if user is stupid and adds tabs 
				//to the registry after start (and calls this constructor via reflection)
				AddTabs(TabControl, monitor.Tabs);
			}

			Content = (Layout) TabControl;
		}

		protected override void InitialiseComponents()
		{
			SaveWindowPosition = true;

			TitleCharacterCasing = CharacterCasing.Normal;

			SetBorderBehaviour(app);

			AddResources();
		}

		/// <summary>
		/// Define how the border of the application behaves.
		/// </summary>
		/// <param name="app">The app environment. </param>
		protected virtual void SetBorderBehaviour(App app)
		{
			//This can only be set in the constructor or onstartup
			BorderThickness = new Thickness(1);
			BorderBrush = UIColours.AccentColorBrush;
			GlowBrush = UIColours.AccentColorBrush;

			//Disable that the titlebar will get grey if not focused. 
			//And any other changes that may occur when the window is not focused.
			NonActiveWindowTitleBrush = UIColours.AccentColorBrush;
			NonActiveBorderBrush = BorderBrush;
			NonActiveGlowBrush = GlowBrush;
		}

		/// <summary>
		/// This function adds all required resources. 
		/// </summary>
		protected virtual void AddResources()
		{

		}

		/// <summary>
		/// THis function creates the <see cref="TabControlUI"/>.
		/// </summary>
		/// <returns>The newly created <see cref="TabControlUI"/>.</returns>
		protected virtual TabControlUI<SigmaWindow> CreateTabControl()
		{
			return new TabControlUI<SigmaWindow>(monitor, app, Title);
		}

		protected virtual TitleBarControl CreateTitleBar()
		{
			TitleBarControl titleBarControl = new TitleBarControl();
			titleBarControl.Margin = new Thickness(0);
			titleBarControl.Padding = new Thickness(0);

			//titleBarControl.AddItem(new TitleBarItem("Environment", "Load", "Store", new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("More", "Extra 3"))));
			//titleBarControl.AddItem(new TitleBarItem("Settings", "Setting 1", "Setting 2"));
			//titleBarControl.AddItem(new TitleBarItem("About", "Sigma"));
			DockPanel dock = new DockPanel();
			dock.Height = 20;

			Menu menu = new Menu();
			menu.IsMainMenu = true;

			DockPanel.SetDock(menu, Dock.Top);


			MenuItem environment = new MenuItem() { Header = "_Environment", VerticalAlignment = VerticalAlignment.Center, FontSize = 12, Height = 28 };
			menu.Items.Add(environment);


			environment.Items.Add(new MenuItem() { Header = "_Save" });
			environment.Items.Add(new MenuItem() { Header = "2" });
			environment.Items.Add(new MenuItem() { Header = "3" });
			environment.Items.Add(new Separator());

			var test = new MenuItem() { Header = "Test" };
			test.Items.Add(new MenuItem() { Header = "Just a prank" });
			environment.Items.Add(test);


			menu.Items.Add(new MenuItem() { Header = "Settings", VerticalAlignment = VerticalAlignment.Center, FontSize = 12, Height = 28 });
			menu.Items.Add(new MenuItem() { Header = "About", VerticalAlignment = VerticalAlignment.Center, FontSize = 12, Height = 28 });

			Style style = new Style(typeof(MenuItem), app.Resources["MaterialDesignMenuItem"] as Style);

			style.Setters.Add(new Setter(BackgroundProperty, Brushes.Transparent));

			Style style2 = new Style(typeof(Menu), (Style)app.Resources["MaterialDesignMenu"]);
			style2.Setters.Add(new Setter(BackgroundProperty, Brushes.Transparent));

			//header.Style = style;

			//app.Resources["MaterialDesignPaper"] = Brushes.Transparent;

			Debug.WriteLine(app.Resources["MaterialDesignPaper"]);

			//dock.Children.Add(menu);

			titleBarControl.Items.Add(menu);

			return titleBarControl;
		}

		/// <summary>
		/// Adds the tabs to the given <see cref="TabControlUI"/>.
		/// </summary>
		/// <param name="tabControl">The <see cref="TabControlUI"/>, where the <see cref="TabItem"/>s will be added to.</param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		protected virtual void AddTabs(TabControlUI<SigmaWindow> tabControl, List<string> names)
		{
			for (int i = 0; i < names.Count; i++)
			{
				tabControl.AddTab(new TabUI(names[i], DefaultGridSize));
			}
		}


	}
}
