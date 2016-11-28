/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using MaterialDesignColors;
using System.Windows;

namespace Sigma.Core.Monitors.WPF.Control.Themes
{
	public interface IColourManager
	{
		/// <summary>
		/// The application environment. 
		/// </summary>
		Application App { get; set; }

		Window @Window { get; set; }

		/// <summary>
		/// The primary colour of the app. Get via <see cref="MaterialDesignValues"/>.
		/// </summary>
		Swatch PrimaryColor { get; set; }

		/// <summary>
		/// The secondary colour of the app. Get via <see cref="MaterialDesignValues"/>.
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
