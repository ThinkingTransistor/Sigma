using System;
using System.Collections.Generic;
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
		/// <summary>
		/// The <see cref="MenuItem"/> behind the <see cref="TitleBarItem"/>.
		/// </summary>
		public MenuItem Content { get; private set; }

		public Dictionary<string, UIElement> Children { get; private set; }

		///// <summary>
		///// The child UIElements that are inside this <see cref="MenuItem"/>.
		///// </summary>
		//public UIElement[] Children { get; private set; }

		/// <summary>
		/// The logiacl parent of this <see cref="TitleBarItem"/>.
		/// </summary>
		public TitleBarItem Parent { get; private set; }

		/// <summary>
		/// Create a <see cref="TitleBarItem"/> with given header and children.
		/// </summary>
		/// <param name="header">The header of the <see cref="TitleBarItem"/>.</param>
		/// <param name="children">The children that will be added. This can be a
		/// <see cref="string"/>, another <see cref="TitleBarItem"/>
		/// or a <see cref="UIElement"/>
		/// - otherwise a <see cref="ArgumentException"/> is thrown.</param>
		public TitleBarItem(object header, params object[] children)
		{
			Content = new MenuItem();
			Content.Header = header;

			Children = new Dictionary<string, UIElement>();
			//Children = new MenuItem[children.Length];

			for (int i = 0; i < children.Length; i++)
			{
				string newElementKey = null;
				UIElement newElement = null;

				if (children[i] is string)
				{
					newElementKey = (string) children[i];
					newElement = new MenuItem() { Header = (string) children[i] };
				}
				else if (children[i] is TitleBarItem)
				{
					TitleBarItem childAsTitleBar = (TitleBarItem) children[i];

					if (childAsTitleBar.Parent != null)
					{
						throw new ArgumentException($"{childAsTitleBar} already has a different parent ({childAsTitleBar.Parent})");
					}

					childAsTitleBar.Parent = this;

					if (childAsTitleBar.Content.Header is string)
					{
						newElementKey = (string) childAsTitleBar.Content.Header;
					}

					newElement = childAsTitleBar.Content;
				}
				else if (children[i] is UIElement)
				{
					newElement = children[i] as UIElement;
				}
				else
				{
					throw new ArgumentException($"{children[i]} is not supported!");
				}

				if (newElementKey == null)
				{
					newElementKey = newElement.ToString();
				}

				Children.Add(newElementKey, newElement);
				Content.Items.Add(newElement);
			
				//Check if next is function
				if (i + 1 < children.Length)
				{
					if (children[i + 1] is Action)
					{
						i++;
						SetFunction(newElement, (Action) children[i]);
					}
				}
			}
		}

		public TitleBarItem SetFunction(string elementKey, Action action)
		{
			return SetFunction(Children[elementKey], action);
		}

		public TitleBarItem SetFunction(UIElement item, Action action)
		{
			if (item == null)
			{
				throw new ArgumentNullException($"{nameof(item)} may not be null!");
			}
			else if (action == null)
			{
				throw new ArgumentNullException($"{nameof(action)} may not be null!");
			}

			Debug.WriteLine($"Added function for {item}");

			item.MouseLeftButtonUp += (sender, args) => action?.Invoke();
			item.TouchDown += (sender, args) => action?.Invoke();

			if (item is MenuItem)
			{
				((MenuItem) item).Click += (sender, args) => action?.Invoke();
			}

			return this;
		}

		public override string ToString()
		{
			return Content.Header.ToString();
		}
	}

}
