/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using MaterialDesignThemes.Wpf;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public abstract class SigmaPanel : Card
	{
		/// <summary>
		/// The title of the Panel
		/// </summary>
		public string Title { get; set; }

		/// <summary>
		/// The style for this panel (since it is
		/// a card the card style will be applied)
		/// </summary>
		private static Style _cardStyle;

		/// <summary>
		/// The panel that is contained by the card.
		/// </summary>
		public readonly DockPanel RootPanel;

		/// <summary>
		/// The Grid Header. Use / modify this if
		/// you want to add icons etc. instead of 
		/// a simple label.
		/// </summary>
		public Grid Header { get; set; }

		/// <summary>
		/// The Grid for the content (only to be
		/// future proof)
		/// </summary>
		private readonly Grid _contentGrid;

		/// <summary>
		/// The content that will be inside the contentGrid.
		/// </summary>
		private UIElement _content;

		/// <summary>
		/// The content that is visible.
		/// </summary>
		public new object Content
		{
			get { return _content; }
			set
			{
				if (_content != null)
				{
					_contentGrid.Children.Remove(_content);
				}

				if (value == null)
				{
					_content = null;
				}
				else
				{
					_content = value as UIElement ?? new Label { Content = value.ToString() };
					_contentGrid.Children.Add(_content);
				}
			}
		}

		/// <summary>
		/// Create a SigmaPanel with a given title. 
		/// If a title is not sufficient modify <see cref="Header"/>.
		/// </summary>
		/// <param name="title">The given tile.</param>
		protected SigmaPanel(string title)
		{
			//Don't do this in static constructor, otherwise
			//we cannot guarantee that the application is already running
			if (_cardStyle == null)
			{
				_cardStyle = Application.Current.Resources[typeof(Card)] as Style;
			}

			Style = _cardStyle;

			Title = title;
			RootPanel = CreateDockPanel();

			Header = CreateHeader();
			AddHeader(RootPanel, Header);

			_contentGrid = CreateContentGrid();
			AddContentGrid(RootPanel, _contentGrid);

			base.Content = RootPanel;
		}

		/// <summary>
		/// Add the content to the grid (may be used for callbacks
		/// etc.)
		/// </summary>
		/// <param name="panel">The root <see cref="UIElement"/> of the panel.</param>
		/// <param name="content">The content that will be added.</param>
		protected virtual void AddContentGrid(DockPanel panel, UIElement content)
		{
			panel.Children.Add(content);
		}

		/// <summary>
		/// Create the header grid and apply the correct theme to it. 
		/// (This could also be done via a custom style) 
		/// </summary>
		/// <returns>The newly created grid. </returns>
		protected virtual Grid CreateHeader()
		{
			Grid header = new Grid();

			header.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
			header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Auto) });

			Label headerContent = new Label { Content = Title };
			header.Children.Add(headerContent);

			header.SetResourceReference(BackgroundProperty, "SigmaPanelHeaderBackground");
			headerContent.SetResourceReference(ForegroundProperty, "SigmaPanelHeaderForeground");

			return header;
		}

		/// <summary>
		/// Add the header the specified panel. 
		/// </summary>
		/// <param name="panel">The root <see cref="UIElement"/> of the panel.</param>
		/// <param name="header">The header to be added.</param>
		protected virtual void AddHeader(DockPanel panel, UIElement header)
		{
			panel.Children.Add(header);
			DockPanel.SetDock(header, Dock.Top);
		}

		/// <summary>
		/// Create the grid where the content is placed in.
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
		/// Create the default panel in which every other element is contained. 
		/// </summary>
		/// <returns>The newly create <see cref="DockPanel"/>.</returns>
		protected virtual DockPanel CreateDockPanel()
		{
			return new DockPanel { LastChildFill = true };
		}
	}
}
