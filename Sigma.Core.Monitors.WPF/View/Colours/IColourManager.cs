/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using MaterialDesignColors;

namespace Sigma.Core.Monitors.WPF.View.Colours
{
	/// <summary>
	/// A <see cref="IColourManager"/> allows to change the look and feel of the application.
	/// </summary>
	public interface IColourManager
	{
		/// <summary>
		///     The application environment.
		/// </summary>
		Application App { get; set; }

		/// <summary>
		/// A reference to the root window this ColourManager belongs to.
		/// </summary>
		Window Window { get; set; }

		/// <summary>
		///     The primary colour of the app. Get via <see cref="Model.UI.Resources.MaterialDesignValues" />.
		/// </summary>
		Swatch PrimaryColor { get; set; }

		/// <summary>
		///     The secondary colour of the app. Get via <see cref="Model.UI.Resources.MaterialDesignValues" />.
		/// </summary>
		Swatch SecondaryColor { get; set; }

		/// <summary>
		///     Switch between light and dark theme.
		/// </summary>
		bool Dark { get; set; }

		/// <summary>
		///     Switch between default and alternate style.
		/// </summary>
		bool Alternate { get; set; }

		/// <summary>
		///     Force an update of all values.
		/// </summary>
		void ForceUpdate();
	}
}