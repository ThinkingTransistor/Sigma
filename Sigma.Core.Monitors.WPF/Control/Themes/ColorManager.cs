/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows;
using MaterialDesignColors;
using MaterialDesignThemes.Wpf;

namespace Sigma.Core.Monitors.WPF.Control.Themes
{
	public class ColorManager : IColorManager
	{
		private Application app;
		private bool appStarted;

		private Swatch primaryColor, secondaryColor;
		private bool dark;

		public ColorManager(Swatch defaultPrimary, Swatch defaultSecondary)
		{
			this.primaryColor = defaultPrimary;
			this.secondaryColor = defaultSecondary;
		}

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
					throw new ArgumentException("App cannot be null!");
				}

				if (value != app)
				{
					app = value;

					appStarted = false;

					if (app != null)
					{
						app.Startup -= AppStartup;
					}

					app.Startup += AppStartup;
				}
			}
		}

		private void AppStartup(object sender, StartupEventArgs e)
		{
			appStarted = true;

			new PaletteHelper().ReplacePrimaryColor(primaryColor);
			new PaletteHelper().ReplaceAccentColor(secondaryColor);
			new PaletteHelper().SetLightDark(dark);
		}

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
					throw new ArgumentException("SecondaryColor cannot be null!");
				}

				primaryColor = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => new PaletteHelper().ReplacePrimaryColor(primaryColor));
				}
			}
		}

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
					throw new ArgumentException("SecondaryColor cannot be null!");
				}

				secondaryColor = value;

				if (appStarted)
				{
					app.Dispatcher.Invoke(() => new PaletteHelper().ReplaceAccentColor(secondaryColor));
				}
			}
		}

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
	}
}
