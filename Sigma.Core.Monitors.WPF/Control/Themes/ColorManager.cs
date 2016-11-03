using System;
using System.Linq;
using System.Windows;
using MahApps.Metro;
using MahApps.Metro.Controls;
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



		//private Application app;
		//private bool appStarted;

		//public Application App
		//{
		//	get
		//	{
		//		return app;
		//	}
		//	set
		//	{
		//		if (value == null)
		//		{
		//			throw new ArgumentException("Application cannot be null");
		//		}

		//		appStarted = false;

		//		//This is not the first app set
		//		if (app != null)
		//		{
		//			app.Startup -= OnStartup;
		//		}
		//		//This is the first app set
		//		else
		//		{
		//			if (Accent == null)
		//			{
		//				Accent = ThemeManager.GetAccent("Blue");
		//			}
		//			if (AppTheme == null)
		//			{
		//				AppTheme = ThemeManager.GetAppTheme("BaseLight");
		//			}
		//		}

		//		app = value;

		//		app.Startup += OnStartup;
		//	}
		//}

		//private Accent accent;

		//public Accent Accent
		//{
		//	get
		//	{
		//		return accent;
		//	}
		//	set
		//	{
		//		if (value == null)
		//		{
		//			throw new ArgumentException("Accent cannot be null");
		//		}

		//		accent = value;

		//		//If the app has already be started, the accent can be directly changed
		//		//otherwise it will be changed automatically on start
		//		if (appStarted)
		//		{
		//			app.BeginInvoke(() => new PaletteHelper().SetLightDark(true));
		//			//new PaletteHelper().SetLightDark(true);
		//			//Console.WriteLine($"U say Nullpointer: Application.Current {Application.Current == null}, accent: {accent == null}, appTheme: {appTheme == null}");
		//			//ThemeManager.ChangeAppStyle(Application.Current, accent, appTheme);
		//		}
		//	}
		//}

		//private AppTheme appTheme;

		//public AppTheme AppTheme
		//{
		//	get
		//	{
		//		return appTheme;
		//	}
		//	set
		//	{
		//		if (value == null)
		//		{
		//			throw new ArgumentException("AppTheme cannot be null");
		//		}

		//		appTheme = value;

		//		//If the app has already be started, the theme can be directly changed
		//		//otherwise it will be changed automatically on start
		//		if (appStarted)
		//		{
		//			//new PaletteHelper().SetLightDark(true);
		//			//ThemeManager.ChangeAppStyle(Application.Current, accent, appTheme);
		//		}
		//	}
		//}

		//private void OnStartup(object sender, StartupEventArgs e)
		//{

		//	//new PaletteHelper().ReplacePrimaryColor("Green");
		//	//ThemeManager.ChangeAppStyle(Application.Current, accent, appTheme);
		//	appStarted = true;
		//}

	}
}
