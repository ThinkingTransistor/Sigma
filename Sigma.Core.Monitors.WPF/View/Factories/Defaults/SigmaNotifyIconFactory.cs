using System;
using System.Drawing;
using System.Windows;
using System.Windows.Forms;
using Application = System.Windows.Application;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public class SigmaNotifyIconFactory : IUIFactory<NotifyIcon>
	{
		private readonly string _title;
		private readonly Icon _icon;
		private readonly EventHandler _doubleClick;
		private readonly MenuItem[] _items;


		public SigmaNotifyIconFactory(string title, string icon, EventHandler doubleClick, MenuItem[] items) : this(title, new Icon(icon), doubleClick, items) { }

		public SigmaNotifyIconFactory(string title, Icon icon, EventHandler doubleClick, MenuItem[] items)
		{
			_title = title;
			_icon = icon;
			_items = items;
			_doubleClick = doubleClick;
		}

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