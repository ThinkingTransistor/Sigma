/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// Every visualiser that is responsible for displaying a parameter (e.g. a CheckBox visualiser ...)
	/// has to implement this interface for easy modification.
	/// 
	/// It also allows the visualiser to be a <see cref="System.Windows.Controls.Control"/> and a <see cref="System.Windows.Controls.UserControl"/>.
	/// </summary>
	public interface IParameterVisualiser
	{
		/// <summary>
		/// The name of the parameter that is being displayed.
		/// </summary>
		string ParameterName { get; set; }

		/// <summary>
		/// Determines whether the parameter is visible or not.
		/// </summary>
		bool IsEnabled { get; set; }

		/// <summary>
		/// Determines whether the parameter is readonly or not. 
		/// </summary>
		bool IsReadOnly { get; set; }
	}
}