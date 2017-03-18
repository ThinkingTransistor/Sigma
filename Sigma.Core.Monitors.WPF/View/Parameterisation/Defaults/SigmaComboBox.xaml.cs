/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows.Controls;
using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.Annotations;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	public partial class SigmaComboBox
	{
		private readonly string[] _keys;
		private readonly object[] _values;

		/// <summary>
		/// The currently selected index. 
		/// </summary>
		public int SelectedIndex { get; set; }

		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

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
		/// This boolean determines whether there are unsaved changes or not.
		/// <c>True</c> if there are other changes, <c>false</c> otherwise.
		/// </summary>
		public override bool Pending { get; set; }

		/// <summary>
		/// This boolean determines whether a synchronisation erroered or not.
		/// <c>True</c> if there are errors, <c>false</c> otherwise.
		/// </summary>
		public override bool Errored { get; set; }

		public SigmaComboBox([NotNull] string[] keys, [NotNull] object[] values)
		{
			if (keys == null) throw new ArgumentNullException(nameof(keys));
			if (values == null) throw new ArgumentNullException(nameof(keys));

			if (keys.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(keys));
			if (values.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(values));

			if (values.Length != keys.Length) throw new ArgumentException("Values require the same length.", nameof(keys));

			_keys = keys;
			_values = values;

			DataContext = this;

			InitializeComponent();

			ComboBox.ItemsSource = keys;
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{

		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			SynchronisationHandler.SynchroniseSet(Registry, Key, (double) _values[SelectedIndex], val => Pending = false, e => Errored = true);
		}

		private void ComboBox_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
		{
			if (SynchronisationHandler != null)
			{
				Write();
			}
		}
	}
}
