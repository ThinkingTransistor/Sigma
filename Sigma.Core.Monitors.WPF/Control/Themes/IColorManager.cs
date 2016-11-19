/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using MaterialDesignColors;

namespace Sigma.Core.Monitors.WPF.Control.Themes
{
	public interface IColorManager
	{
		/// <summary>
		/// The application environment. 
		/// </summary>
		Application App { get; set; }

		Window @Window { get; set; }

		/// <summary>
		/// The primary colour of the app. Get via <see cref="MaterialDesignSwatches"/>.
		/// </summary>
		Swatch PrimaryColor { get; set; }

		/// <summary>
		/// The secondary colour of the app. Get via <see cref="MaterialDesignSwatches"/>.
		/// </summary>
		Swatch SecondaryColor { get; set; }

		/// <summary>
		/// Switch between light and dark theme.
		/// </summary>
		bool Dark { get; set; }

		/// <summary>
		/// Switch between default and alternate style. 
		/// </summary>
		bool Alternate { get; set; }
	}
}
