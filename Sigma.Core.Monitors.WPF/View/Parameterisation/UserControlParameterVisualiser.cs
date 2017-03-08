/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// This class is a wrapper or <see cref="IParameterVisualiser"/>. It automatically implements
	/// code that is global for all <see cref="UserControl"/>s.
	/// 
	/// When extending from this class you have to call InitializeComponent manually.
	/// </summary>
	public abstract class UserControlParameterVisualiser : UserControl, IParameterVisualiser
	{
		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public abstract bool IsReadOnly { get; set; }

		/// <summary>
		/// The name of the parameter that is being displayed.
		/// </summary> 
		public abstract string ParameterName { get; set; }
	}
}
