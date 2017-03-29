/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Drawing;
using System.IO;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Resources;
using Application = System.Windows.Application;

// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	/// <summary>
	/// This class is responsible for generating a <see cref="NotifyIcon"/>. This can be used to push notifications, and enable 
	/// the app to be run in the background (with this little icon in the lower right corner). 
	/// </summary>
	public class SigmaNotifyIconFactory : IUIFactory<NotifyIcon>
	{
		/// <summary>
		/// The title of the <see cref="NotifyIcon"/> - visible when hovering. 
		/// </summary>
		private string _title;
		/// <summary>
		/// The icon of the <see cref="NotifyIcon"/> - permanently visible. 
		/// </summary>
		private Icon _icon;
		/// <summary>
		/// The action that will be invoked when double clicking the icon - normally this opens the app. 
		/// </summary>
		private EventHandler _doubleClick;
		/// <summary>
		/// The <see cref="MenuItem"/>s that will be available when right-clicking the <see cref="NotifyIcon"/>.
		/// </summary>
		private MenuItem[] _items;

		protected SigmaNotifyIconFactory() { }

		public SigmaNotifyIconFactory(string title, string iconResource, EventHandler doubleClick, MenuItem[] items)
		{
			Init(title, iconResource, doubleClick, items);
		}

		public SigmaNotifyIconFactory(string title, Icon icon, EventHandler doubleClick, MenuItem[] items)
		{
			Init(title, icon, doubleClick, items);
		}

		/// <summary>
		/// Initialise the passed values. (Simply set the correct fields of the class).
		/// </summary>
		/// <param name="title"><see cref="_title"/></param>
		/// <param name="iconResource"><see cref="_icon"/></param>
		/// <param name="doubleClick"><see cref="_doubleClick"/></param>
		/// <param name="items"><see cref="_items"/></param>
		protected virtual void Init(string title, string iconResource, EventHandler doubleClick, MenuItem[] items)
		{
			StreamResourceInfo streamResourceInfo = Application.GetResourceStream(new Uri(iconResource));
			if (streamResourceInfo == null)
			{
				throw new ArgumentException($@"Could not create a resource stream to {nameof(iconResource)}: {iconResource}. Is it really a resource in the format: pack://application:,,,/YourReferencedAssembly;component/YourPossibleSubFolder/YourResourceFile.ico",
					nameof(iconResource));
			}

			using (Stream iconStream = streamResourceInfo.Stream)
			{
				Init(title, new Icon(iconStream), doubleClick, items);
			}
		}

		/// <summary>
		/// Initialise the passed values. (Simply set the correct fields of the class).
		/// </summary>
		/// <param name="title"><see cref="_title"/></param>
		/// <param name="icon"><see cref="_icon"/></param>
		/// <param name="doubleClick"><see cref="_doubleClick"/></param>
		/// <param name="items"><see cref="_items"/></param>
		protected virtual void Init(string title, Icon icon, EventHandler doubleClick, MenuItem[] items)
		{
			_title = title;
			_icon = icon;
			_items = items;
			_doubleClick = doubleClick;
		}

		/// <summary>
		/// Create a <see cref="NotifyIcon"/> and set all required parameters (normally it is an <see cref="UIElement"/>). 
		/// If additional parameters are required, use <see cref="parameters"/>.
		/// </summary>
		/// <param name="app">The <see cref="Application"/> in which the newly generated item will be.</param>
		/// <param name="window">The <see cref="Window"/> in which the newly generated item will be.</param>
		/// <param name="parameters">The parameters that may or may not be required. Often none are required.</param>
		/// <returns>The newly created item.</returns>
		public NotifyIcon CreateElement(Application app, Window window, params object[] parameters)
		{
			NotifyIcon notify = new NotifyIcon
			{
				Text = _title,
				Icon = _icon,
				Visible = true,
			};

			notify.DoubleClick += _doubleClick;
			notify.ContextMenu = new ContextMenu(_items);

			return notify;
		}
	}
}