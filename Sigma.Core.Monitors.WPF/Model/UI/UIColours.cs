using System.Windows;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.Model.UI
{
	public static class UIColours
	{
		public static FontFamily @FontFamily { get; private set; }

		public static Brush HighlightColorBrush { get; private set; }

		/// <summary>
		/// This is per default a bluish colour. This is equal to Primary500. 
		/// </summary>
		public static Brush AccentColorBrush { get; private set; }
		public static Brush AccentColorBrush2 { get; private set; }
		public static Brush AccentColorBrush3 { get; private set; }
		public static Brush AccentColorBrush4 { get; private set; }

		public static Brush AccentSelectedColorBrush { get; private set; }

		#region Alias

		public static Brush IdealForegroundColorBrush { get; private set; }
		public static Brush WindowTitleColorBrush { get; private set; }
		public static Brush IdealForegroundDisabledBrush { get; private set; }

		#endregion


		static UIColours()
		{
			Application app = Application.Current;

			FontFamily = app.Resources["MaterialDesignFont"] as FontFamily;

			HighlightColorBrush = app.Resources["HighlightBrush"] as Brush;

			AccentColorBrush = app.Resources["AccentColorBrush"] as Brush;

			AccentColorBrush2 = app.Resources["AccentColorBrush2"] as Brush;

			AccentColorBrush3 = app.Resources["AccentColorBrush3"] as Brush;

			AccentColorBrush4 = app.Resources["AccentColorBrush4"] as Brush;

			AccentSelectedColorBrush = app.Resources["AccentSelectedColorBrush"] as Brush;


			//Alias references
			WindowTitleColorBrush = HighlightColorBrush;
			IdealForegroundColorBrush = AccentSelectedColorBrush;
			IdealForegroundDisabledBrush = AccentColorBrush;
		}
	}
}
