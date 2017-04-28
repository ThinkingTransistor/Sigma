/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
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
		/// <summary>
		/// The object that is currently being displayed (without updating the displayed information)
		/// </summary>
		protected object _Object;

		/// <summary>
		/// The object that is being displayed (toString is called).
		/// </summary>
		public virtual object Object
		{
			get { return _Object; }
			set
			{
				_Object = value;
				string text = value?.ToString() ?? "null";
				TextBlock.Text = Prefix + text + Postfix;
			}
		}

		/// <summary>
		/// The text that is visualised. 
		/// </summary>
		public string Text => TextBlock.Text;

		/// <summary>
		/// This string will be added before the displayed string.
		/// </summary>
		public string Prefix
		{
			get { return (string) GetValue(PrefixProperty); }
			set { SetValue(PrefixProperty, value); }
		}

		public static readonly DependencyProperty PrefixProperty =
			DependencyProperty.Register("Prefix", typeof(string), typeof(SigmaTextBox), new PropertyMetadata(""));

		/// <summary>
		/// This string will be added after the displayed string.
		/// </summary>
		public string Postfix
		{
			get { return (string) GetValue(PostfixProperty); }
			set { SetValue(PostfixProperty, value); }
		}

		public static readonly DependencyProperty PostfixProperty =
			DependencyProperty.Register("Postfix", typeof(string), typeof(SigmaTextBox), new PropertyMetadata(""));

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
			SynchronisationHandler.SynchroniseUpdate(Registry, Key, Object, newObj => Dispatcher.Invoke(() => Object = newObj));
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
