/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Windows;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.Panels
{
	public abstract class SigmaPanel : Card
	{
		/// <summary>
		///     The style for this panel (since it is
		///     a card the card style will be applied)
		/// </summary>
		private static Style _cardStyle;

		/// <summary>
		///     The Grid for the content (only to be
		///     future proof)
		/// </summary>
		protected readonly Grid ContentGrid;

		/// <summary>
		///     The panel that is contained by the card.
		/// </summary>
		public readonly DockPanel RootPanel;

		/// <summary>
		///     The content that will be inside the contentGrid.
		/// </summary>
		private UIElement _content;

		/// <summary>
		/// Currently responsible monitor - it will be automatically set when adding a new panel. (<c>null</c> until <see cref="Initialise"/>)
		/// </summary>
		public WPFMonitor Monitor { get; set; }

		/// <summary>
		/// The default loading indicator factory that will be used for new panels.
		/// </summary>
		public static IUIFactory<UIElement> DefaultLoadingIndicatorFactory = View.Factories.Defaults.LoadingIndicatorFactory.Factory;

		/// <summary>
		/// The loading indicator. It will only be generated if <see cref="UseLoadingIndicator"/> is true while adding the panel.
		/// </summary>
		private UIElement _loadingIndicator;

		/// <summary>
		/// Determines whether a loading indactor is shown or not.
		/// </summary>
		protected bool UseLoadingIndicator;

		/// <summary>
		/// Determines whether the loading indicator is visible or not.
		/// </summary>
		protected bool LoadingIndicatorVisible { get; private set; }

		/// <summary>
		/// The factory that will be used to generate a loading indicator.
		/// </summary>
		protected IUIFactory<UIElement> LoadingIndicatorFactory;

		/// <summary>
		///     Create a SigmaPanel with a given title.
		///     If a title is not sufficient modify <see cref="Header" />.
		/// </summary>
		/// <param name="title">The given tile.</param>
		/// <param name="content">The content for the header. If <c>null</c> is passed,
		/// the title will be used.</param>
		protected SigmaPanel(string title, object content = null)
		{
			//Don't do this in static constructor, otherwise
			//we cannot guarantee that the application is already running
			if (_cardStyle == null)
			{
				_cardStyle = Application.Current.Resources[typeof(Card)] as Style;
			}

			Style = _cardStyle;

			LoadingIndicatorFactory = DefaultLoadingIndicatorFactory;

			Title = title;
			RootPanel = CreateDockPanel();

			Header = CreateHeader(content ?? title);
			AddHeader(RootPanel, Header);

			ContentGrid = CreateContentGrid();
			AddContentGrid(RootPanel, ContentGrid);

			base.Content = RootPanel;
		}

		/// <summary>
		///     The title of the Panel
		/// </summary>
		public string Title { get; set; }

		/// <summary>
		///     The Grid Header. Use / modify this if
		///     you want to add icons etc. instead of
		///     a simple label.
		/// </summary>
		public Grid Header { get; set; }

		/// <summary>
		///     The content that is visible.
		/// </summary>
		public new object Content
		{
			get { return _content; }
			set
			{
				if (_content != null)
				{
					ContentGrid.Children.Remove(_content);
				}

				if (value == null)
				{
					_content = null;
				}
				else
				{
					_content = value as UIElement ?? new Label { Content = value.ToString() };
					ContentGrid.Children.Add(_content);
				}
			}
		}

		#region CardProperties

		/// <summary>
		///     Gets or sets the outer margin of an element.
		///     <returns>
		///         Provides margin values for the element. The default value is a <see cref="Thickness" />
		///         with all properties equal to 0 (zero).
		///     </returns>
		/// </summary>
		public new Thickness Margin
		{
			get { return ContentGrid.Margin; }
			set { ContentGrid.Margin = value; }
		}

		#endregion

		/// <summary>
		///     Add the content to the grid (may be used for callbacks
		///     etc.)
		/// </summary>
		/// <param name="panel">The root <see cref="UIElement" /> of the panel.</param>
		/// <param name="content">The content that will be added.</param>
		protected virtual void AddContentGrid(DockPanel panel, UIElement content)
		{
			panel.Children.Add(content);
		}

		/// <summary>
		///     Create the header grid and apply the correct theme to it.
		///     (This could also be done via a custom style)
		/// </summary>
		/// <param name="content">The content for the header. (Set the content of a label).</param>
		/// <returns>The newly created grid. </returns>
		protected virtual Grid CreateHeader(object content)
		{
			Grid header = new Grid();

			header.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
			header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Auto) });

			Label headerContent = new Label { Content = content };
			header.Children.Add(headerContent);

			header.SetResourceReference(BackgroundProperty, "SigmaPanelHeaderBackground");
			headerContent.SetResourceReference(ForegroundProperty, "SigmaPanelHeaderForeground");

			return header;
		}

		/// <summary>
		///     Add the header the specified panel.
		/// </summary>
		/// <param name="panel">The root <see cref="UIElement" /> of the panel.</param>
		/// <param name="header">The header to be added.</param>
		protected virtual void AddHeader(DockPanel panel, UIElement header)
		{
			panel.Children.Add(header);
			DockPanel.SetDock(header, Dock.Top);
		}

		/// <summary>
		///     Create the grid where the content is placed in.
		/// </summary>
		/// <returns>The newly created grid. </returns>
		protected virtual Grid CreateContentGrid()
		{
			Grid grid = new Grid();

			grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
			grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });

			return grid;
		}

		/// <summary>
		/// This method invokes the initialisation of the panel (after it has been addded).
		/// </summary>
		public void Initialise(WPFWindow window)
		{
			if (UseLoadingIndicator)
			{
				_loadingIndicator = LoadingIndicatorFactory.CreateElement(window.App, window, this);
				ShowLoadingIndicator();
			}
			LoadingIndicatorFactory = null;

			if (window is SigmaWindow)
			{
				OnInitialise((SigmaWindow)window);
			}

			OnInitialise(window);
		}

		/// <summary>
		/// This method shows the loading indicator if not visible.
		/// </summary>
		protected void ShowLoadingIndicator()
		{
			if (_loadingIndicator != null)
			{
				if (!LoadingIndicatorVisible)
				{
					ContentGrid.Dispatcher.Invoke(() =>ContentGrid.Children.Add(_loadingIndicator));
					LoadingIndicatorVisible = true;
				}
			}
		}

		/// <summary>
		/// This method hides the loading indicator if visible.
		/// </summary>
		protected void HideLoadingIndicator()
		{
			if (_loadingIndicator != null)
			{
				if (LoadingIndicatorVisible)
				{
					ContentGrid.Dispatcher.Invoke(() => ContentGrid.Children.Remove(_loadingIndicator));
					LoadingIndicatorVisible = false;
				}
			}
		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected virtual void OnInitialise(WPFWindow window)
		{

		}

		/// <summary>
		/// This method will be called once the window is initialising (after it has been added).
		/// Do not store a reference of the window unless you properly dispose it (remove reference once not required).
		/// </summary>
		/// <param name="window">The wpf window this panel will be added to.</param>
		protected virtual void OnInitialise(SigmaWindow window)
		{

		}

		/// <summary>
		///     Create the default panel in which every other element is contained.
		/// </summary>
		/// <returns>The newly create <see cref="DockPanel" />.</returns>
		protected virtual DockPanel CreateDockPanel()
		{
			return new DockPanel { LastChildFill = true, Margin = new Thickness(-1, 0, 0, 0) };
		}
	}
}