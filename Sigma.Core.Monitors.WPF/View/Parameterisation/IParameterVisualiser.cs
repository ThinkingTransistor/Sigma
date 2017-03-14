/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.Synchronisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// Every visualiser that is responsible for displaying a parameter (e.g. a CheckBox visualiser ...)
	/// has to implement this interface for easy modification.
	/// 
	/// Every visualiser requires an <b>empty</b> constructor.
	/// 
	/// It also allows the visualiser to be a <see cref="System.Windows.Controls.Control"/> and a <see cref="System.Windows.Controls.UserControl"/>.
	/// </summary>
	public interface IParameterVisualiser
	{
		/// <summary>
		/// Determines whether the parameter is visible or not.
		/// </summary>
		bool IsEnabled { get; set; }

		/// <summary>
		/// Determines whether the parameter is readonly or not. 
		/// </summary>
		bool IsReadOnly { get; set; }

		/// <summary>
		/// The fully resolved key to access the synchandler.
		/// </summary>
		string Key { get; set; }

		/// <summary>
		/// The SynchronisationHandler that is used to sync the parameter with the training process.
		/// </summary>
		ISynchronisationHandler SynchronisationHandler { get; set; }
	}
}