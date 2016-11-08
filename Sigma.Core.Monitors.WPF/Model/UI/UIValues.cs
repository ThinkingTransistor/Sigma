using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;

namespace Sigma.Core.Monitors.WPF.Model.UI
{
	public static class UIValues
	{
		public static FontFamily @FontFamily { get; private set; }

		/// <summary>
		/// This is per default a bluish colour.
		/// </summary>
		public static Brush AccentColorBrush { get; private set; }

		static UIValues()
		{
			Application app = Application.Current;

			FontFamily = app.Resources["MaterialDesignFont"] as FontFamily;
			AccentColorBrush = app.Resources["AccentColorBrush"] as Brush;
		}
	}
}
