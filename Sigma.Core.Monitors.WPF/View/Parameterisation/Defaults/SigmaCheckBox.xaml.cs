/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

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
		private bool _isChecked;

		/// <summary>
		/// Determines whether the UserControl is checked or not. 
		/// </summary>
		public bool IsChecked
		{
			get { return _isChecked; }
			set
			{
				_isChecked = value;
				Write();
			}
		}

		/// <summary>
		/// This boolean determines whether there are unsaved changes or not.
		/// <c>True</c> if there are other changes, <c>false</c> otherwise.
		/// </summary>
		public override bool Pending { get; set; }

		/// <summary>
		/// This boolean determines whether a synchronisation erroered or not.
		/// <c>True</c> if there are errors, <c>false</c> otherwise.
		/// </summary>
		public override bool Errored { get; set; }

		/// <summary>
		/// The default constructor for a <see cref="SigmaCheckBox"/>.
		/// </summary>
		public SigmaCheckBox()
		{
			InitializeComponent();

			DataContext = this;
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			IsChecked = SynchronisationHandler.SynchroniseGet<bool>(Registry, Key);
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			Pending = true;
			SynchronisationHandler.SynchroniseSet(Registry, Key, IsChecked, val => Pending = false, e => Errored = true);
		}
	}
}
