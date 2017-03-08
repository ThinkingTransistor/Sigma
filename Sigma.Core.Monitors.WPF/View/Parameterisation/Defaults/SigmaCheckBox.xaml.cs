/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// Interaction logic for SigmaCheckBox.xaml
	/// </summary>
	public partial class SigmaCheckBox : IParameterVisualiser
	{
		/// <summary>
		/// Determines whether the parameter is edible or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

		/// <summary>
		/// The name of the parameter that is being displayed.
		/// </summary>
		public override string ParameterName
		{
			get { return ParameterNameTextBlock.Text; }
			set { ParameterNameTextBlock.Text = value; }
		}

		/// <summary>
		/// Determines whether the UserControl is checked or not. 
		/// </summary>
		public bool IsChecked { get; set; }

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
