using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// Every visualiser that is responsible for displaying a parameter (e.g. a CheckBox visualiser ...)
	/// has to implement this interface for easy modification.
	/// 
	/// It also allows the visualiser to be a <see cref="Control"/> and a <see cref="UserControl"/>.
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
		/// Determines whether the parameter is edible or not. 
		/// </summary>
		bool IsReadOnly { get; set; }
	}


}