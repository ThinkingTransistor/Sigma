using System.Collections.Generic;
using MahApps.Metro.Controls;
using Sigma.Core.Monitors.WPF.View.TitleBar;

namespace Sigma.Core.Monitors.WPF.Control.TitleBar
{
	public class TitleBarControl : WindowCommands
	{
		/// <summary>
		/// The children of the <see cref="TitleBarControl"/>.
		/// </summary>
		private List<TitleBarItem> children;

		public TitleBarControl()
		{
			children = new List<TitleBarItem>();

			//Styling options
			ShowLastSeparator = false;
		}

		/// <summary>
		/// Add a <see cref="TitleBarItem"/> to the <see cref="TitleBarControl"/>.
		/// Do not use <see cref="ItemCollection.Add"/>. (Although it will be called internally) 
		/// </summary>
		/// <param name="item">The item to add.</param>
		public void AddItem(TitleBarItem item)
		{
			Items.Add(item.Content);
			children.Add(item);
		}

		/// <summary>
		/// Remove a <see cref="TitleBarItem"/> from the <see cref="TitleBarControl"/>.
		/// </summary>
		/// <param name="item"></param>
		public void RemoveItem(TitleBarItem item)
		{
			Items.Remove(item.Content);
			children.Remove(item);
		}

		/// <summary>
		/// Get a child at the specified index.
		/// </summary>
		/// <param name="i">The specified index. </param>
		/// <returns></returns>
		public TitleBarItem this[int i]
		{
			get { return children[i]; }
		}

		//#region DependencyProperties

		//public static readonly DependencyProperty ChildrenProperty = DependencyProperty.Register("Text", typeof(string), typeof(TitleBarItem), new UIPropertyMetadata("null"));

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
	}
}
