/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Windows.Input;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	[ParameterVisualiser(typeof(float), Priority = VisualiserPriority.Lower)]
	internal class SigmaFloatBox : SigmaNumberBox<float> { }

	[ParameterVisualiser(typeof(double), Priority = VisualiserPriority.Lower)]
	internal class SigmaDoubleBox : SigmaNumberBox<double> { }


	/// <summary>
	/// Sigmas way of displaying numbers. 
	/// </summary>
	internal class SigmaNumberBox<T> : SigmaTextBox
	{
		/// <summary>
		/// The converter that converts the given type for the registry
		/// </summary>
		protected readonly TypeConverter Converter;

		internal SigmaNumberBox()
		{
			Converter = TypeDescriptor.GetConverter(typeof(T));
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			Text = SynchronisationHandler.SynchroniseGet<T>(Registry, Key).ToString();
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			try
			{
				T num = (T) Converter.ConvertFromString(Text);
				Pending = true;
				SynchronisationHandler.SynchroniseSet(Registry, Key, num, val => Pending = false, e => Errored = true);
			}
			catch (Exception)
			{
				Errored = true;
			}
		}
	}

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
		public string Text
		{
			get { return TextBox.Text; }
			set { TextBox.Text = value; }
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
		/// Create a new default textbox that can display parameters.
		/// </summary>
		public SigmaTextBox()
		{
			InitializeComponent();

			DataContext = this;
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			Text = SynchronisationHandler.SynchroniseGet<string>(Registry, Key);
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			Pending = true;
			SynchronisationHandler.SynchroniseSet(Registry, Key, Text, val => Pending = false, e => Errored = true);
		}

		/// <summary>
		/// This method is executed when a keydown is detected. If enter is detected, the registry is written.
		/// </summary>
		/// <param name="sender">The sender of the event.</param>
		/// <param name="e">The arguments for the event.</param>
		private void OnKeyDownHandler(object sender, KeyEventArgs e)
		{
			if (e.Key == System.Windows.Input.Key.Enter)
			{
				Write();
			}
		}
	}
}
