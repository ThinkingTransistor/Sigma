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
		/// Create a new default textbox that can display parameters.
		/// </summary>
		public SigmaTextBox()
		{
			InitializeComponent();

			DataContext = this;
		}
	}
}
