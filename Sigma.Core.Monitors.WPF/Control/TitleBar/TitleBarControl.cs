/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections;
using System.Collections.Generic;
using System.Windows.Controls;
using MahApps.Metro.Controls;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.View.TitleBar;

namespace Sigma.Core.Monitors.WPF.Control.TitleBar
{
	public class TitleBarControl : WindowCommands, IEnumerable<TitleBarItem>
	{
		/// <summary>
		/// The children of the <see cref="TitleBarControl"/>.
		/// </summary>
		private readonly Dictionary<string, TitleBarItem> _children;

		public Menu @Menu { get; }

		public TitleBarControl()
		{
			_children = new Dictionary<string, TitleBarItem>();

			Menu = new Menu();

			Items.Add(Menu);

			//Styling options
			ShowLastSeparator = false;
		}

		/// <summary>
		/// Add a <see cref="TitleBarItem"/> to the <see cref="TitleBarControl"/>.
		/// Do not use <see cref="ItemCollection.Add"/> ore <see cref="Menu.Items.Add"/>. (Although it will be called internally) 
		/// </summary>
		/// <param name="item">The item to add.</param>
		/// <param name="applyColor">This boolean decides whether the foreground colour should be changed to white.
		/// (Recommended for headings)</param>
		public void AddItem(TitleBarItem item, bool applyColor = true)
		{
			Menu.Items.Add(item.Content);
			_children.Add(item.ToString(), item);

			if (applyColor)
			{
				item.Content.Foreground = UiResources.IdealForegroundColorBrush;
			}
		}

		/// <summary>
		/// Remove a <see cref="TitleBarItem"/> from the <see cref="TitleBarControl"/>.
		/// </summary>
		/// <param name="item"></param>
		public void RemoveItem(TitleBarItem item)
		{
			Menu.Items.Remove(item.Content);
			_children.Remove(item.ToString());
		}

		public IEnumerator<TitleBarItem> GetEnumerator()
		{
			return _children.Values.GetEnumerator();
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return _children.Values.GetEnumerator();
		}

		/// <summary>
		/// Get a child at the specified index.
		/// </summary>
		/// <param name="str">The specified index. </param>
		/// <returns></returns>
		public TitleBarItem this[string str] => _children[str];
	}
}
