/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.ViewModel.TitleBar
{
	public class TitleBarItem
	{
		public List<TitleBarItem> TitleBarItemChildren;

		/// <summary>
		///     Create a <see cref="TitleBarItem" /> with given header and children.
		/// </summary>
		/// <param name="header">The header of the <see cref="TitleBarItem" />.</param>
		/// <param name="children">
		///     The children that will be added. A child can be a
		///     <see cref="string" />, another <see cref="TitleBarItem" />
		///     or an <see cref="UIElement" />
		///     - otherwise a <see cref="ArgumentException" /> is thrown.
		/// </param>
		public TitleBarItem(object header, params object[] children) : this(new MenuItem { Header = header }, children)
		{
		}

		/// <summary>
		///     Create a <see cref="TitleBarItem" /> with given <see cref="MenuItem" /> and children.
		/// </summary>
		/// <param name="item">
		///     The item that will become the <see cref="Content" /> of this
		///     <see cref="TitleBarItem" />.
		/// </param>
		/// <param name="children">
		///     The children that will be added. A child can be a
		///     <see cref="string" />, another <see cref="TitleBarItem" />
		///     or an <see cref="UIElement" />
		///     - otherwise a <see cref="ArgumentException" /> is thrown.
		/// </param>
		public TitleBarItem(MenuItem item, params object[] children)
		{
			Content = item;
			Children = new Dictionary<string, UIElement>();
			TitleBarItemChildren = new List<TitleBarItem>();

			// if the first parameter is for the given item
			// this is out of the loop in order to improve performance
			int startIndex = 0;
			if (children.Length > 0)
			{
				if (TryAddFunction(item, children[0]))
				{
					startIndex = 1;
				}
			}

			for (int i = startIndex; i < children.Length; i++)
			{
				string newElementKey = null;
				UIElement newElement;

				if (children[i] is string)
				{
					//The key and header are simply the string
					newElementKey = (string) children[i];
					newElement = new MenuItem { Header = (string) children[i] };
				}
				else if (children[i] is TitleBarItem)
				{
					TitleBarItem childAsTitleBar = (TitleBarItem) children[i];
					TitleBarItemChildren.Add(childAsTitleBar);

					if (childAsTitleBar.Parent != null)
					{
						throw new ArgumentException($"{childAsTitleBar} has already a different parent ({childAsTitleBar.Parent})");
					}

					childAsTitleBar.Parent = this;

					//TODO: validate if
					//not required because the ToString of a string is the string
					//if (childAsTitleBar.Content.Header is string)
					//{
					//	newElementKey = (string) childAsTitleBar.Content.Header;
					//}

					newElement = childAsTitleBar.Content;
				}
				else if (children[i] is UIElement)
				{
					newElement = children[i] as UIElement;
				}
				else
				{
					throw new ArgumentException($"{children[i]} with the type {children[i].GetType()} is not supported!");
				}

				if (newElementKey == null)
				{
					newElementKey = newElement.ToString();
				}

				//Add the content to the dictionary and the
				//UIElement itself
				Children.Add(newElementKey, newElement);
				Content.Items.Add(newElement);

				//Check if next parameter is a function
				if (i + 1 < children.Length && TryAddFunction(newElement, children[i + 1]))
				{
					i++;
				}
			}
		}

		public Application App { get; set; }
		public Window Window { get; set; }

		/// <summary>
		///     The <see cref="MenuItem" /> behind the <see cref="TitleBarItem" />.
		/// </summary>
		public MenuItem Content { get; }

		/// <summary>
		///     The child UIElements that are inside this <see cref="MenuItem" />.
		/// </summary>
		public Dictionary<string, UIElement> Children { get; }

		/// <summary>
		///     The logical parent of this <see cref="TitleBarItem" />.
		/// </summary>
		public TitleBarItem Parent { get; private set; }

		/// <summary>
		/// This method trys to cast the object to an <see cref="Action"/> - 
		/// if it is, it will be added and true is returned.
		/// </summary>
		/// <param name="item">The item that the function will be added to.</param>
		/// <param name="function">The (maybe) function that will be added</param>
		/// <returns><c>True</c>if function is a suitable function. <c>False</c> otherwise. </returns>
		private bool TryAddFunction(UIElement item, object function)
		{
			if (function is Action)
			{
				SetFunction(item, (Action) function);
			}
			else if (function is Action<Application, Window, TitleBarItem>)
			{
				SetFunction(item, (Action<Application, Window, TitleBarItem>) function);
			}
			else
			{
				return false;
			}

			return true;
		}

		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />. This
		///     element has to be identifiable by its <paramref name="elementKey" />.
		///     If this is not the case see <see cref="SetFunction(UIElement, Action)" />
		///     or <see cref="Children" />.
		/// </summary>
		/// <param name="elementKey">
		///     The identifier for the object the <see cref="Action" />
		///     will be assigned to - in most cases, the <see cref="object.ToString" />
		///     method of the object.
		/// </param>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		protected TitleBarItem SetFunction(string elementKey, Action action)
		{
			return SetFunction(Children[elementKey], action);
		}

		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />. This
		///     element has to be identifiable by its <paramref name="elementKey" />.
		///     If this is not the case see <see cref="SetFunction(UIElement, Action)" />
		///     or <see cref="Children" />.
		/// </summary>
		/// <param name="elementKey">
		///     The identifier for the object the <see cref="Action" />
		///     will be assigned to - in most cases, the <see cref="object.ToString" />
		///     method of the object.
		/// </param>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		protected TitleBarItem SetFunction(string elementKey, Action<Application, Window, TitleBarItem> action)
		{
			return SetFunction(Children[elementKey], action);
		}

		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />.
		///     Use this function for speed or when the key may be
		///     ambiguous.
		/// </summary>
		/// <param name="item">
		///     The object the <see cref="Action" />
		///     will be assigned to.
		/// </param>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		protected TitleBarItem SetFunction(UIElement item, Action action)
		{
			return SetFunction(item, (app, window, element) => action());
		}

		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />.
		///     Use this function for speed or when the key may be
		///     ambiguous.
		/// </summary>
		/// <param name="item">
		///     The object the <see cref="Action" />
		///     will be assigned to.
		/// </param>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		protected TitleBarItem SetFunction(UIElement item, Action<Application, Window, TitleBarItem> action)
		{
			if (item == null)
			{
				throw new ArgumentNullException(nameof(item));
			}
			if (action == null)
			{
				throw new ArgumentNullException(nameof(action));
			}

			item.MouseLeftButtonUp += (sender, args) => action(App, Window, this);
			item.TouchDown += (sender, args) => action(App, Window, this);

			MenuItem menuItem = item as MenuItem;
			if (menuItem != null)
			{
				menuItem.Click += (sender, args) => action(App, Window, this);
			}

			return this;
		}

		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />.
		///     Use this function for speed or when the key may be
		///     ambiguous.
		/// </summary>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		public TitleBarItem SetFunction(Action<Application, Window, TitleBarItem> action)
		{
			return SetFunction(Content, action);
		}


		/// <summary>
		///     Set the onClick for a given <see cref="UIElement" />.
		///     Use this function for speed or when the key may be
		///     ambiguous.
		/// </summary>
		/// <param name="action">
		///     The <see cref="Action" /> that will be assigned
		///     to the <see cref="UIElement" />
		/// </param>
		/// <returns>The <see cref="TitleBarItem" /> for concatenation. </returns>
		public TitleBarItem SetFunction(Action action)
		{
			return SetFunction(Content, action);
		}

		public override string ToString()
		{
			return Content.Header.ToString();
		}
	}
}
