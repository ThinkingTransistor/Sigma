/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// A custom checkbox that implements <see cref="IParameterVisualiser"/> and can visualise values from a registry
	/// (booleans in this case).
	/// </summary>
	[ParameterVisualiser(typeof(bool), Priority = VisualiserPriority.Lower)]
	public partial class SigmaCheckBox
	{
		/// <summary>
		/// Determines whether the parameter is edible or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

		/// <summary>
		/// Determines whether the UserControl is checked or not. 
		/// </summary>
		public bool IsChecked { get; set; }

		/// <summary>
		/// The fully resolved key to access the synchandler.
		/// </summary>
		public override string Key { get; set; }

		/// <summary>
		/// The default constructor for a <see cref="SigmaCheckBox"/>.
		/// </summary>
		public SigmaCheckBox()
		{
			InitializeComponent();

			DataContext = this;
		}
	}
}
