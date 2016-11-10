using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Model.UI;
using System;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

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
					newElement = new Button { Content = contents[i], FontFamily = UIColours.FontFamily };
				}
				else
				{
					throw new ArgumentException($"Unsupported object {contents[i]}. Supported types: TitleBarItem, PopupBox, string, and UIElement");
				}

				contentPanel.Children.Add(newElement);

				Elements[i] = newElement;
			}

			Content.PopupContent = contentPanel;
		}

		private void PrepareChild(TitleBarItem child)
		{
			child.parent = this;
		}

		private UIElement ModifyChild(PopupBox child)
		{
			child.PlacementMode = PopupBoxPlacementMode.RightAndAlignMiddles;

			//TODO: fixme
			child.VerticalAlignment = VerticalAlignment.Center;
			child.HorizontalAlignment = HorizontalAlignment.Center;
			child.Height = 36;
			child.Width = double.NaN;

			//child.MouseEnter += (a,b) => child.ToggleContentTemplate. = UIValues.AccentColorBrush;

			//var button =  new Button() { Content = child.ToggleContent};

			//child.ToggleContent = null;

			//button.MouseLeftButtonDown += (sender, args) =>
			//{
			//	Debug.WriteLine($"Hover - content: {child.PopupContent}");
			//	child.IsPopupOpen = true;
			//};

			//return button;

			return child;
		}

	}
}
