namespace Sigma.Core.Monitors.WPF.View.Parameterisation
{

	/// <summary>
	/// Interaction logic for SigmaCheckBox.xaml
	/// </summary>
	public partial class SigmaCheckBox 
	{
		/// <summary>
		/// Determines whether the parameter is edible or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

		/// <summary>
		/// The name of the parameter that is being displayed.
		/// </summary>
		public override string ParameterName {
			get { return ParameterNameBox.Text; }
			set { ParameterNameBox.Text = value; }
		}
	}
}
