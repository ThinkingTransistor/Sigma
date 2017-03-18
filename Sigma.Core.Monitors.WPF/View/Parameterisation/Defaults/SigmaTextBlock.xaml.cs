/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// Sigmas way of displaying objects. 
	/// </summary>
	[ParameterVisualiser(typeof(object), Priority = VisualiserPriority.Lower)]
	public partial class SigmaTextBlock
	{
		private object _object;

		/// <summary>
		/// The object that is being displayed (toString is called).
		/// </summary>
		public object Object
		{
			get { return _object; }
			set
			{
				_object = value;
				TextBlock.Text = value?.ToString() ?? "null";
			}
		}

		/// <summary>
		/// The text that is visualised. 
		/// </summary>
		public string Text => TextBlock.Text;


		/// <summary>
		/// The fully resolved key to access the synchandler.
		/// </summary>
		public override string Key { get; set; }

		/// <summary>
		/// The registry for which the visualiser displays values. (e.g. operators registry)
		/// </summary>
		public override IRegistry Registry { get; set; }

		/// <summary>
		/// The SynchronisationHandler that is used to sync the parameter with the training process.
		/// </summary>
		public override ISynchronisationHandler SynchronisationHandler { get; set; }

		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; } = true;

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
		/// Create a new default textblock that can display parameters (i.e. objects).
		/// </summary>
		public SigmaTextBlock()
		{
			InitializeComponent();

			DataContext = this;
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			Object = SynchronisationHandler.SynchroniseGet<object>(Registry, Key);
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			Pending = true;
			SynchronisationHandler.SynchroniseSet(Registry, Key, Object, val => Pending = false, e => Errored = true);
		}
	}
}
