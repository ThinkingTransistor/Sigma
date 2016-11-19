/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.Model.UI.Resources
{
	public static class UiResources
	{
		#region Font
		public static FontFamily @FontFamily { get; private set; }

		#region FontSize
		public static readonly double H1;
		public static readonly double H2;
		public static readonly double H3;
		public static readonly double H4;
		public static readonly double H5;
		public static readonly double H6;

		public static readonly double P1;
		public static readonly double P2;
		public static readonly double P3;

		#endregion FontSize

		public static Style MaterialDesignDisplay4TextBlock { get; private set; }
		public static Style MaterialDesignDisplay3TextBlock { get; private set; }
		public static Style MaterialDesignDisplay2TextBlock { get; private set; }
		public static Style MaterialDesignDisplay1TextBlock { get; private set; }

		public static Style MaterialDesignHeadlineTextBlock { get; private set; }
		public static Style MaterialDesignTitleTextBlock { get; private set; }
		public static Style MaterialDesignSubheadingTextBlock { get; private set; }
		public static Style MaterialDesignBody2TextBlock { get; private set; }
		public static Style MaterialDesignBody1TextBlock { get; private set; }
		public static Style MaterialDesignCaptionTextBlock { get; private set; }
		public static Style MaterialDesignButtonTextBlock { get; private set; }

		#region Hyperlinks
		public static Style MaterialDesignDisplay4Hyperlink { get; private set; }
		public static Style MaterialDesignDisplay3Hyperlink { get; private set; }
		public static Style MaterialDesignDisplay2Hyperlink { get; private set; }
		public static Style MaterialDesignDisplay1Hyperlink { get; private set; }

		public static Style MaterialDesignHeadlineHyperlink { get; private set; }
		public static Style MaterialDesignTitleHyperlink { get; private set; }
		public static Style MaterialDesignSubheadingHyperlink { get; private set; }
		public static Style MaterialDesignBody2Hyperlink { get; private set; }
		public static Style MaterialDesignBody1Hyperlink { get; private set; }
		public static Style MaterialDesignCaptionHyperlink { get; private set; }
		#endregion
		#endregion

		#region Colour
		public static Brush HighlightColorBrush { get; private set; }

		/// <summary>
		/// This is per default a bluish colour. This is equal to Primary500. 
		/// It should be used for most UI content. 
		/// </summary>
		public static Brush AccentColorBrush { get; private set; }
		public static Brush AccentColorBrush2 { get; private set; }
		public static Brush AccentColorBrush3 { get; private set; }
		public static Brush AccentColorBrush4 { get; private set; }

		public static Brush AccentSelectedColorBrush { get; private set; }

		public static Brush MaterialDesignBody { get; private set; }
		public static Brush MaterialDesignBodyLight { get; private set; }

		#region Alias

		public static Brush IdealForegroundColorBrush { get; private set; }
		public static Brush WindowTitleColorBrush { get; private set; }
		public static Brush IdealForegroundDisabledBrush { get; private set; }

		#endregion
		#endregion

		static UiResources ()
		{
			Application app = Application.Current;

			#region Font
			FontFamily = (FontFamily) app.Resources["MaterialDesignFont"];

			H1 = (double) app.Resources["H1"];
			H2 = (double) app.Resources["H2"];
			H3 = (double) app.Resources["H3"];
			H4 = (double) app.Resources["H4"];
			H5 = (double) app.Resources["H5"];
			H6 = (double) app.Resources["H6"];

			P1 = (double) app.Resources["P1"];
			P2 = (double) app.Resources["P2"];
			P3 = (double) app.Resources["P3"];


			MaterialDesignDisplay4TextBlock = (Style) app.Resources["MaterialDesignDisplay4TextBlock"];
			MaterialDesignDisplay3TextBlock = (Style) app.Resources["MaterialDesignDisplay3TextBlock"];
			MaterialDesignDisplay2TextBlock = (Style) app.Resources["MaterialDesignDisplay2TextBlock"];
			MaterialDesignDisplay1TextBlock = (Style) app.Resources["MaterialDesignDisplay1TextBlock"];

			MaterialDesignHeadlineTextBlock = (Style) app.Resources["MaterialDesignHeadlineTextBlock"];
			MaterialDesignTitleTextBlock = (Style) app.Resources["MaterialDesignTitleTextBlock"];
			MaterialDesignSubheadingTextBlock = (Style) app.Resources["MaterialDesignSubheadingTextBlock"];
			MaterialDesignBody2TextBlock = (Style) app.Resources["MaterialDesignBody2TextBlock"];
			MaterialDesignBody1TextBlock = (Style) app.Resources["MaterialDesignBody1TextBlock"];
			MaterialDesignCaptionTextBlock = (Style) app.Resources["MaterialDesignCaptionTextBlock"];
			MaterialDesignButtonTextBlock = (Style) app.Resources["MaterialDesignButtonTextBlock"];

			#region Hyperlinks
			MaterialDesignDisplay4Hyperlink = (Style) app.Resources["MaterialDesignDisplay4Hyperlink"];
			MaterialDesignDisplay3Hyperlink = (Style) app.Resources["MaterialDesignDisplay3Hyperlink"];
			MaterialDesignDisplay2Hyperlink = (Style) app.Resources["MaterialDesignDisplay2Hyperlink"];
			MaterialDesignDisplay1Hyperlink = (Style) app.Resources["MaterialDesignDisplay1Hyperlink"];

			MaterialDesignHeadlineHyperlink = (Style) app.Resources["MaterialDesignHeadlineHyperlink"];
			MaterialDesignTitleHyperlink = (Style) app.Resources["MaterialDesignTitleHyperlink"];
			MaterialDesignSubheadingHyperlink = (Style) app.Resources["MaterialDesignSubheadingHyperlink"];
			MaterialDesignBody2Hyperlink = (Style) app.Resources["MaterialDesignBody2Hyperlink"];
			MaterialDesignBody1Hyperlink = (Style) app.Resources["MaterialDesignBody1Hyperlink"];
			MaterialDesignCaptionHyperlink = (Style) app.Resources["MaterialDesignCaptionHyperlink"];
			#endregion Hyperlinks
			#endregion Font

			#region Colour
			HighlightColorBrush = (Brush) app.Resources["HighlightBrush"];

			AccentColorBrush = (Brush) app.Resources["AccentColorBrush"];
			AccentColorBrush2 = (Brush) app.Resources["AccentColorBrush2"];
			AccentColorBrush3 = (Brush) app.Resources["AccentColorBrush3"];
			AccentColorBrush4 = (Brush) app.Resources["AccentColorBrush4"];

			AccentSelectedColorBrush = (Brush) app.Resources["AccentSelectedColorBrush"];

			MaterialDesignBody = (Brush) app.Resources["MaterialDesignBody"];
			MaterialDesignBodyLight = (Brush) app.Resources["MaterialDesignBodyLight"];

			//Alias references
			WindowTitleColorBrush = HighlightColorBrush;
			IdealForegroundColorBrush = AccentSelectedColorBrush;
			IdealForegroundDisabledBrush = AccentColorBrush;

			#endregion Colour
		}
	}
}
