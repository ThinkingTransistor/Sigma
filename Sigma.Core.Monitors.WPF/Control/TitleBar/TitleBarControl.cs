/* 
MIT License

Copyright (c) 2016 Florian C�sar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections;
using System.Collections.Generic;
using System.Windows.Controls;
using MahApps.Metro.Controls;
using Sigma.Core.Monitors.WPF.View.TitleBar;

namespace Sigma.Core.Monitors.WPF.Control.TitleBar
{
	public class TitleBarControl : WindowCommands, IEnumerable<TitleBarItem>
	{
		/// <summary>
		/// The children of the <see cref="TitleBarControl"/>.
		/// </summary>
		private Dictionary<string, TitleBarItem> children;

		public Menu @Menu { get; private set; }

		public TitleBarControl()
		{
			children = new Dictionary<string, TitleBarItem>();

			Menu = new Menu();
			//Menu.FontSize = UIResources.P1;

			//Menu.Background = Brushes.Transparent;
			Items.Add(Menu);

			//Styling options
			ShowLastSeparator = false;
		}

		/// <summary>
		/// Add a <see cref="TitleBarItem"/> to the <see cref="TitleBarControl"/>.
		/// Do not use <see cref="ItemCollection.Add"/> ore <see cref="Menu.Items.Add"/>. (Although it will be called internally) 
		/// </summary>
		/// <param name="item">The item to add.</param>
		public void AddItem(TitleBarItem item)
		{
			Menu.Items.Add(item.Content);
			children.Add(item.ToString(), item);
		}

		/// <summary>
		/// Remove a <see cref="TitleBarItem"/> from the <see cref="TitleBarControl"/>.
		/// </summary>
		/// <param name="item"></param>
		public void RemoveItem(TitleBarItem item)
		{
			Menu.Items.Remove(item.Content);
			children.Remove(item.ToString());
		}

		public IEnumerator<TitleBarItem> GetEnumerator()
		{
			return children.Values.GetEnumerator();
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return children.Values.GetEnumerator();
		}

		/// <summary>
		/// Get a child at the specified index.
		/// </summary>
		/// <param name="i">The specified index. </param>
		/// <returns></returns>
		public TitleBarItem this[string str]
		{
			get { return children[str]; }
		}
	}
}
