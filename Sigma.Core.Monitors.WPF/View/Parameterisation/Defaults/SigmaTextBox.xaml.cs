namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	[ParameterVisualiser(typeof(string), Priority = ParameterVisualiserAttribute.VisualiserPriority.Lower)]
	[ParameterVisualiser(typeof(object), Priority = ParameterVisualiserAttribute.VisualiserPriority.Lower)]
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

		public SigmaTextBox()
		{
			InitializeComponent();

			DataContext = this;
		}
	}
}
