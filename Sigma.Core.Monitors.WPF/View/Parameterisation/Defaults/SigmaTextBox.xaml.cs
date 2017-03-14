/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// Sigmas way of displaying strings. 
	/// </summary>
	[ParameterVisualiser(typeof(string), Priority = VisualiserPriority.Lower)]
	public partial class SigmaTextBox
	{
		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

		/// <summary>
		/// The text that is visualised. 
		/// </summary>
		public string Text { get; set; }

		/// <summary>
		/// The fully resolved key to access the synchandler.
		/// </summary>
		public override string Key { get; set; }

		/// <summary>
		/// Create a new default textbox that can display parameters.
		/// </summary>
		public SigmaTextBox()
		{
			InitializeComponent();

			DataContext = this;
		}
	}
}
