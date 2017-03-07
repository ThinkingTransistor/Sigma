using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{
	/// <summary>
	/// This class is a wrapper or <see cref="IParameterVisualiser"/>. It automatically implements
	/// code that is global for all <see cref="UserControl"/>s.
	/// </summary>
	public abstract partial class UserControlParameterVisualiser : IParameterVisualiser
	{
		/// <summary>
		/// The default constructor for the <see cref="UserControlParameterVisualiser"/>.
		/// </summary>
		protected UserControlParameterVisualiser()
		{
			InitializeComponent();
		}

		/// <summary>
		/// Determines whether the parameter is edible or not. 
		/// </summary>
		public abstract bool IsReadOnly { get; set; }

		/// <summary>
		/// The name of the parameter that is being displayed.
		/// </summary>
		public abstract string ParameterName { get; set; }
	}
}
