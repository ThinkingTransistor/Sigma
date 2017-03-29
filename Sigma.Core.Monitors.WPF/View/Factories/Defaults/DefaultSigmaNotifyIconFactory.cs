/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows.Forms;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	public sealed class DefaultSigmaNotifyIconFactory : SigmaNotifyIconFactory
	{
		public DefaultSigmaNotifyIconFactory(string iconResource, Action maximise, Action forceClose)
		{
			MenuItem[] items = new MenuItem[2];

			items[0] = new MenuItem(Properties.Resources.OpenApp) { DefaultItem = true };
			items[0].Click += (sender, args) => maximise();

			items[1] = new MenuItem(Properties.Resources.CloseApp);
			items[1].Click += (sender, args) => forceClose();

			Init("Sigma", iconResource, (sender, args) => maximise(), items);
		}
	}
}