/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.View.CustomControls.TitleBar;
using Sigma.Core.Monitors.WPF.View.Windows;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	/// <summary>
	/// This <see cref="IUIFactory{T}"/> creates a <see cref="SigmaAboutBox"/> with specified text and <see cref="DialogHost"/>.
	/// 
	/// The style will be applied from the file: SigmaAboutBoxStyle.xaml
	/// </summary>
	public class AboutFactory : IUIFactory<UIElement>
	{
		/// <summary>
		/// The <see cref="DialogHost"/> that will be used (e.g. to close it again). 
		/// </summary>
		private readonly DialogHost _windowDialogHost;

		/// <summary>
		/// Create a new <see cref="AboutFactory"/> with the <see cref="DialogHost"/> from <see cref="SigmaWindow"/>.
		/// </summary>
		/// <param name="window">The <see cref="SigmaWindow"/>, the <see cref="AboutFactory"/> will produce in. </param>
		public AboutFactory(SigmaWindow window) : this(window.DialogHost) { }

		/// <summary>
		/// Create a new <see cref="AboutFactory"/> with a given <see cref="DialogHost"/>.
		/// </summary>
		/// <param name="windowDialogHost">The <see cref="DialogHost"/>, that will be passed to the box. </param>
		public AboutFactory(DialogHost windowDialogHost)
		{
			_windowDialogHost = windowDialogHost;
		}

		public UIElement CreateElement(Application app, Window window, params object[] parameters)
		{
			return new SigmaAboutBox
			{
				DialogHost = _windowDialogHost,
				Heading = "Sigma",
				Text = "Rocket powered machine learning.\n" +
						"Create, compare, adapt, improve - neural networks at the speed of thought.\n" +
						"Free to use for anyone (MIT license)."
			};
		}
	}
}