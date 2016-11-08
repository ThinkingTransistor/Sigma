using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Dragablz.Dockablz;
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
			FontFamily = UIValues.FontFamily;

			TitleAlignment = HorizontalAlignment.Center;

			TitleBar = new TitleBarControl();
			TitleBar.AddItem(new TitleBarItem("Environment", "Load", "Store", new TitleBarItem("Extras", "Extra1", "Extra2", new TitleBarItem("Extra Extra", "Extra 3"))));
			TitleBar.AddItem(new TitleBarItem("Settings", "Setting 1", "Setting 2"));
			TitleBar.AddItem(new TitleBarItem("About", "Sigma"));

			LeftWindowCommands = TitleBar;

			//LeftWindowCommands = new TitleBarControl();
			//LeftWindowCommands.Items.Add(new TitleBarItem("Environment"));
			//LeftWindowCommands.Items.Add(new TitleBarItem("Settings"));
			//LeftWindowCommands.Items.Add(new TitleBarItem("About"));

			//StackPanel stackPanel = new StackPanel();
			//stackPanel.Children.Add(new Button { Content = "Load" });
			//stackPanel.Children.Add(new Button { Content = "Save" });


			////StackPanel innerPanel = new StackPanel();
			////innerPanel.Children.Add(new Button() { Content = "Option1" });
			////innerPanel.Children.Add(new Button() { Content = "Option2" });

			////PopupBox innerBox = new PopupBox();
			////innerBox.PopupContent = innerPanel;
			////innerBox.ToggleContent = "Extras";
			////innerBox.VerticalAlignment = VerticalAlignment.Center;
			////innerBox.HorizontalAlignment = HorizontalAlignment.Center;
			////innerBox.Style = new Button().Style;
			////innerBox.PlacementMode = PopupBoxPlacementMode.RightAndAlignMiddles;
			////stackPanel.Children.Add(innerBox);

			//PopupBox popupBox = new PopupBox();
			//popupBox.PopupContent = stackPanel;
			//popupBox.ToggleContent = "Environment";
			//popupBox.StaysOpen = true;

			

			//popupBox.PopupMode = PopupBoxPopupMode.MouseOverEager; 

			//LeftWindowCommands.Items.Add(popupBox);


			//ToolBarTray tray = new ToolBarTray();
			//ToolBar toolbar = new ToolBar();
			//((IAddChild)tray).AddChild(toolbar);

			//toolbar.Items.Add(new Button() { Content = "New" });


			//LeftWindowCommands = new MahApps.Metro.Controls.WindowCommands();
			//LeftWindowCommands.Items.Add(tray);



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

		/// <summary>
		/// Adds the tabs to the given <see cref="TabControlUI"/>.
		/// </summary>
		/// <param name="tabControl">The <see cref="TabControlUI"/>, where the <see cref="TabItem"/>s will be added to.</param>
		/// <param name="names">A list that contains the names of each tab that will be created. </param>
		protected virtual void AddTabs(TabControlUI<SigmaWindow> tabControl, List<string> names)
		{
			for (int i = 0; i < names.Count; i++)
			{
				tabControl.AddTab(new TabUI(names[i]));
			}
		}


	}
}
