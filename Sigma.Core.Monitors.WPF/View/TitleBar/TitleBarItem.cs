using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Media;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Model.UI;

namespace Sigma.Core.Monitors.WPF.View.TitleBar
{
	public class TitleBarItem
	{
		//#region DependencyProperties

		//public static readonly DependencyProperty TextProperty = DependencyProperty.Register("Text", typeof(string), typeof(TitleBarItem), new UIPropertyMetadata("null"));

		//#endregion DependencyProperties

		//#region Properties

		///// <summary>
		///// The text that is displayed for the <see cref="TitleBarItem"/>.
		///// </summary>
		//public string Text
		//{
		//	get { return (string) GetValue(TextProperty); }
		//	set { SetValue(TextProperty, value); }
		//}

		//#endregion Properties

		public PopupBox Content { get; private set; }

		private StackPanel contentPanel;

		public UIElement[] Elements { get; set; }

		private TitleBarItem parent;

		public TitleBarItem(object toggleContent, params object[] contents)
		{
			Content = new PopupBox() { StaysOpen = true };


			if (toggleContent != null)
			{
				Content.ToggleContent = toggleContent;
				//TODO: Color and font
				//Content.ToggleContentTemplate

				Debug.WriteLine($"Content: {Content.ToggleContent.GetType()}");
			}

			Elements = new UIElement[contents.Length];

			contentPanel = new StackPanel();

			for (int i = 0; i < contents.Length; i++)
			{
				UIElement newElement = null;
				if (contents[i] is TitleBarItem)
				{
					TitleBarItem child = (TitleBarItem) contents[i];
					PrepareChild(child);

					newElement = ModifyChild(child.Content);
				}
				else if (contents[i] is PopupBox)
				{
					newElement = ModifyChild((PopupBox) contents[i]);
				}
				else if (contents[i] is UIElement)
				{
					newElement = (UIElement) contents[i];
				}
				else if (contents[i] is string)
				{
					Debug.WriteLine($"String: {contents[i] as string}");
					newElement = new Button { Content = contents[i], FontFamily = UIValues.FontFamily };
				}
				else
				{
					throw new ArgumentException($"Unsupported object {contents[i]}. Supported types: TitleBarItem, PopupBox, string, and UIElement");
				}

				contentPanel.Children.Add(newElement);

				Elements[i] = newElement;
			}

			Content.PopupContent = contentPanel;

			//Background = Brushes.Transparent;
			//BorderBrush = Brushes.Transparent;
			//FontSize = 15;

			//Style = Application.Current.FindResource("WindowCommandsPopupBoxStyle") as Style;
			//StackPanel panel = new StackPanel();
			//panel.Children.Add(new Button() { Content = text });
			//base.Content = panel;
		}

		private void PrepareChild(TitleBarItem child)
		{
			child.parent = this;
		}

		private PopupBox ModifyChild(PopupBox child)
		{
			child.PlacementMode = PopupBoxPlacementMode.RightAndAlignMiddles;

			return child;
		}
	}
}
