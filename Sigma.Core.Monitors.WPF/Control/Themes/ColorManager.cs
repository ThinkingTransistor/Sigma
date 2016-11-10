/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Dragablz;
using MaterialDesignColors;
using MaterialDesignThemes.Wpf;
using System;
using System.Windows;

namespace Sigma.Core.Monitors.WPF.Control.Themes
{
	public class ColorManager : IColorManager
	{
		/// <summary>
		/// The application environment.
		/// </summary>
		private Application app;
		/// <summary>
		/// Tells whether onStartup has already been called on the app.
		/// </summary>
		private bool appStarted;

		/// <summary>
		/// The primary colour.
		/// </summary>
		private Swatch primaryColor;
		/// <summary>
		/// The secondary colour.
		/// </summary>
		private Swatch secondaryColor;
		/// <summary>
		/// Whether a dark or a light theme should be applied.
		/// </summary>
		private bool dark;
		/// <summary>
		/// Option for an alternate style for tabs.
		/// </summary>
		private bool alternate;

		/// <summary>
		/// Create a new <see cref="ColorManager"/>.
		/// </summary>
		/// <param name="defaultPrimary">The default primary colour (if none has been set).</param>
		/// <param name="defaultSecondary">The default secondary colour (if none has been set).</param>
		public ColorManager(Swatch defaultPrimary, Swatch defaultSecondary)
		{
			this.primaryColor = defaultPrimary;
			this.secondaryColor = defaultSecondary;
		}

		/// <summary>
		/// The application environment. 
		/// </summary>
		public Application App
		{
			get
			{
				return app;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("App cannot be null!");
				}

				//If the value has changed
				if (value != app)
				{
					app = value;

					//reset appstarted
					appStarted = false;

					if (app != null)
					{
						app.Startup -= AppStartup;
					}

					app.Startup += AppStartup;
				}
			}
		}

		/// <summary>
		/// The event listener that will be added to the app. 
		/// </summary>
		/// <param name="sender">The sender of the event.</param>
		/// <param name="e">The <see cref="StartupEventArgs"/>.</param>
		private void AppStartup(object sender, StartupEventArgs e)
		{
			appStarted = true;

			new PaletteHelper().ReplacePrimaryColor(primaryColor);
			new PaletteHelper().ReplaceAccentColor(secondaryColor);
			new PaletteHelper().SetLightDark(dark);
			ApplyStyle(alternate);
		}

		/// <summary>
		/// The primary colour of the app. Get via <see cref="MaterialDesignSwatches"/>.
		/// </summary>
		public Swatch PrimaryColor
		{
			get
			{
				return primaryColor;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("SecondaryColor cannot be null!");
				}

				primaryColor = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => new PaletteHelper().ReplacePrimaryColor(primaryColor));
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
				return secondaryColor;
			}

			set
			{
				if (value == null)
				{
					throw new ArgumentNullException("SecondaryColor cannot be null!");
				}

				secondaryColor = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => new PaletteHelper().ReplaceAccentColor(secondaryColor));
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
				return dark;
			}
			set
			{
				dark = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => new PaletteHelper().SetLightDark(dark));
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
				return alternate;
			}

			set
			{
				alternate = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => ApplyStyle(alternate));
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
			var style = app.TryFindResource(styleKey) as Style;

			App.Resources[typeof(TabablzControl)] = style;
		}
	}
}
