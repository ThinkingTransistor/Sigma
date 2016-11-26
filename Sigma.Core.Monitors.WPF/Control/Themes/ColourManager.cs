/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Dragablz;
using MaterialDesignColors;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.View.Windows;
// ReSharper disable AccessToStaticMemberViaDerivedType

namespace Sigma.Core.Monitors.WPF.Control.Themes
{
	public class ColourManager : IColourManager
	{
		/// <summary>
		/// The application environment.
		/// </summary>
		private Application _app;

		/// <summary>
		/// The corresponding <see cref="SigmaWindow"/>.
		/// </summary>
		private SigmaWindow _sigmaWindow;

		public Window Window
		{
			get { return _sigmaWindow; }
			set
			{
				_sigmaWindow = value as SigmaWindow;
			}
		}

		/// <summary>
		/// Tells whether onStartup has already been called on the app.
		/// </summary>
		private bool _appStarted;

		/// <summary>
		/// The primary colour.
		/// </summary>
		private Swatch _primaryColor;
		/// <summary>
		/// The secondary colour.
		/// </summary>
		private Swatch _secondaryColor;
		/// <summary>
		/// Whether a dark or a light theme should be applied.
		/// </summary>
		private bool _dark;

		/// <summary>
		/// Option for an alternate style for tabs.
		/// </summary>
		private bool _alternate;

		/// <summary>
		/// The path for the Sigma light style theme.
		/// </summary>
		private const string SIGMA_STYLE_LIGHT_PATH = "pack://application:,,,/Sigma.Core.Monitors.WPF;component/View/Styles/LightStyle.xaml";
		/// <summary>
		/// The path for the Sigma dark style theme.
		/// </summary>
		private const string SIGMA_STYLE_DARK_PATH = "pack://application:,,,/Sigma.Core.Monitors.WPF;component/View/Styles/DarkStyle.xaml";

		/// <summary>
		/// The resource dictionary for the light theme.
		/// This parameter is null until the application started event
		/// has been called.
		/// </summary>
		private ResourceDictionary _sigmaStyleLightDictionary;
		/// <summary>
		/// The resource dictionary for the dark theme.
		/// This parameter is null until the application started event
		/// has been called.
		/// </summary>
		private ResourceDictionary _sigmaStyleDarkDictionary;

		/// <summary>
		/// Create a new <see cref="ColourManager"/>.
		/// </summary>
		/// <param name="defaultPrimary">The default primary colour (if none has been set).</param>
		/// <param name="defaultSecondary">The default secondary colour (if none has been set).</param>
		public ColourManager(Swatch defaultPrimary, Swatch defaultSecondary)
		{
			_primaryColor = defaultPrimary;
			_secondaryColor = defaultSecondary;
		}

		/// <summary>
		/// The application environment. 
		/// </summary>
		public Application App
		{

			get
			{
				return _app;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("App cannot be null!");
				}

				//If the value has not changed
				if (value == _app) return;

				_app = value;

				//reset the values
				_appStarted = false;

				if (_app != null)
				{
					_app.Startup -= AppStartup;
				}

				_app.Startup += AppStartup;
			}
		}

		/// <summary>
		/// The event listener that will be added to the app. 
		/// </summary>
		/// <param name="sender">The sender of the event.</param>
		/// <param name="e">The <see cref="StartupEventArgs"/>.</param>
		private void AppStartup(object sender, StartupEventArgs e)
		{
			_sigmaStyleLightDictionary = new ResourceDictionary { Source = new Uri(SIGMA_STYLE_LIGHT_PATH, UriKind.Absolute) };
			_sigmaStyleDarkDictionary = new ResourceDictionary { Source = new Uri(SIGMA_STYLE_DARK_PATH, UriKind.Absolute) };

			_appStarted = true;

			ReplacePrimaryColor(_primaryColor);
			ReplaceSecondaryColor(_secondaryColor);
			SetLightDark(_dark);

			ApplyStyle(_alternate);
		}

		/// <summary>
		/// Change the theme to dark or light depending on the given parameter.
		/// </summary>
		/// <param name="dark">If this value is <c>true</c>, a dark theme will be applied.
		/// Otherwise a light theme. </param>
		private void SetLightDark(bool dark)
		{
			new PaletteHelper().SetLightDark(dark);

			LoadNewSigmaStyle(dark);

			//Fix the MenuBar
			var correctBrush = (dark
				? UIResources.IdealForegroundColorBrush
				: Application.Current.Resources["SigmaMenuItemForegroundLight"]) as Brush;

			Style newStyle = new Style(typeof(MenuItem), Application.Current.Resources["SigmaMaterialDesignMenuItem"] as Style);
			newStyle.Setters.Add(new Setter(MenuItem.ForegroundProperty, correctBrush));
			Application.Current.Resources[typeof(MenuItem)] = newStyle;
		}

		/// <summary>
		/// Load the new sigma style depending on the passed boolean.
		/// </summary>
		/// <param name="dark">If this value is <c>true</c>, a dark theme will be applied.
		/// Otherwise a light theme. </param>
		private void LoadNewSigmaStyle(bool dark)
		{
			ResourceDictionary oldResourceDictionary = dark ? _sigmaStyleLightDictionary : _sigmaStyleDarkDictionary;
			ResourceDictionary newResourceDictionary = dark ? _sigmaStyleDarkDictionary : _sigmaStyleLightDictionary;

			if (!App.Resources.MergedDictionaries.Contains(newResourceDictionary))
			{
				App.Resources.MergedDictionaries.Add(newResourceDictionary);
			}

			if (App.Resources.MergedDictionaries.Contains(oldResourceDictionary))
			{
				App.Resources.MergedDictionaries.Remove(oldResourceDictionary);
			}
		}

		/// <summary>
		/// Replace the secondary colour.
		/// </summary>
		/// <param name="secondaryColor">The specified <see cref="Swatch"/> that will
		/// be the new secondary colour.</param>
		private static void ReplaceSecondaryColor(Swatch secondaryColor)
		{
			new PaletteHelper().ReplaceAccentColor(secondaryColor);
		}

		/// <summary>
		/// Replace the primary colour.
		/// </summary>
		/// <param name="primaryColor">The specified <see cref="Swatch"/> that will
		/// be the new primary colour.</param>
		private static void ReplacePrimaryColor(Swatch primaryColor)
		{
			new PaletteHelper().ReplacePrimaryColor(primaryColor);
		}

		/// <summary>
		/// The primary colour of the app. Get via <see cref="MaterialDesignSwatches"/>.
		/// </summary>
		public Swatch PrimaryColor
		{
			get
			{
				return _primaryColor;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("SecondaryColor cannot be null!");
				}

				_primaryColor = value;

				if (_appStarted)
				{
					_app.Dispatcher.Invoke(() => new PaletteHelper().ReplacePrimaryColor(_primaryColor));
				}
			}
		}

		/// <summary>
		/// The secondary colour of the app. Get via <see cref="MaterialDesignSwatches"/>.
		/// </summary>
		public Swatch SecondaryColor
		{
			get
			{
				return _secondaryColor;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("SecondaryColor cannot be null!");
				}

				_secondaryColor = value;

				if (_appStarted)
				{
					_app.Dispatcher.Invoke(() => new PaletteHelper().ReplaceAccentColor(_secondaryColor));
				}
			}
		}

		/// <summary>
		/// Switch between light and dark theme.
		/// </summary>
		public bool Dark
		{
			get
			{
				return _dark;
			}
			set
			{
				_dark = value;

				if (_appStarted)
				{
					_app.Dispatcher.Invoke(() => SetLightDark(_dark));
				}
			}
		}

		/// <summary>
		/// Switch between default and alternate style (especially for tabs).
		/// </summary>
		public bool Alternate
		{
			get
			{
				return _alternate;
			}

			set
			{
				_alternate = value;

				if (_appStarted)
				{
					_app.Dispatcher.Invoke(() => ApplyStyle(_alternate));
				}
			}
		}

		/// <summary>
		/// Change the style to normal or alternate.
		/// </summary>
		/// <param name="alternate">Decides which style should be applied.</param>
		private void ApplyStyle(bool alternate)
		{
			var styleKey = alternate ? "MaterialDesignAlternateTabablzControlStyle" : "MaterialDesignTabablzControlStyle";
			var style = _app.TryFindResource(styleKey) as Style;

			App.Resources[typeof(TabablzControl)] = style;
		}
	}
}
