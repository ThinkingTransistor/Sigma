using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Panels
{
	public abstract class SigmaPanel : Card
	{
		public string Title { get; set; }

		private static Style _cardStyle;

		private readonly DockPanel _rootPanel;
		public Panel Header { get; set; }

		private UIElement _content;
		public new UIElement Content
		{
			get { return _content; }
			set
			{
				if (_content != null)
				{
					_rootPanel.Children.Remove(_content);
				}

				_content = value;

				if (_content != null)
				{
					_rootPanel.Children.Add(_content);
				}
			}
		}

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

			_rootPanel = CreateDockPanel();
			Header = CreateHeader();

			AddItems(_rootPanel, Header);

			base.Content = _rootPanel;
		}

		protected virtual Panel CreateHeader()
		{
			Grid header = new Grid();

			header.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Auto) });
			header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Auto) });

			header.Children.Add(new Label
			{
				Content = Title
			});


			return header;
		}

		protected virtual void AddItems(DockPanel panel, UIElement header)
		{
			panel.Children.Add(header);
			DockPanel.SetDock(header, Dock.Top);
		}

		protected virtual DockPanel CreateDockPanel()
		{
			return new DockPanel { LastChildFill = true };
		}
	}
}
