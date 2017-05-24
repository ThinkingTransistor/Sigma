/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults
{
	/// <summary>
	/// A factory that can create a loading indicator.
	/// </summary>
	public class LoadingIndicatorFactory : IUIFactory<UIElement>
	{
		private static readonly LoadingIndicatorFactory _factory;
		/// <summary>
		/// The singleton of the factory.
		/// </summary>
		public static LoadingIndicatorFactory Factory => _factory;

		static LoadingIndicatorFactory()
		{
			_factory = new LoadingIndicatorFactory();
		}

		private LoadingIndicatorFactory() { }

		/// <inheritdoc />
		public UIElement CreateElement(Application app, Window window, params object[] parameters)
		{
			StackPanel panel = new StackPanel
			{
				VerticalAlignment = VerticalAlignment.Center,
				HorizontalAlignment = HorizontalAlignment.Center
			};

			ProgressBar loading = new ProgressBar
			{
				Value = 0,
				IsIndeterminate = true,
				Style = Application.Current.Resources["MaterialDesignCircularProgressBar"] as Style
			};

			loading.Width = loading.Height = 128;

			panel.Children.Add(loading);

			return panel;
		}
	}
}