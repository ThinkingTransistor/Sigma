/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Resources;
using System.Windows.Input;
using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	//[ParameterVisualiser(typeof(float), Priority = VisualiserPriority.Lower)]
	//internal class SigmaFloatBox : DynamicConverterBox
	//{ }

	//[ParameterVisualiser(typeof(double), Priority = VisualiserPriority.Lower)]
	//internal class SigmaDoubleBox : DynamicConverterBox
	//{ }

	///// <summary>
	///// Sigmas way of converting a string to given value (e.g. double)
	///// </summary>
	//public class SigmaConverterBox<T> : SigmaTextBox
	//{
	//	/// <summary>
	//	/// The converter that converts the given type for the registry
	//	/// </summary>
	//	protected readonly TypeConverter Converter;

	//	public SigmaConverterBox()
	//	{
	//		Converter = TypeDescriptor.GetConverter(typeof(T));
	//	}

	//	/// <summary>
	//	/// Force the visualiser to update its value (i.e. display the value that is stored).
	//	/// </summary>
	//	public override void Read()
	//	{
	//		Text = SynchronisationHandler.SynchroniseGet<T>(Registry, Key).ToString();
	//	}

	//	/// <summary>
	//	/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
	//	/// </summary>
	//	public override void Write()
	//	{
	//		try
	//		{
	//			T num = (T) Converter.ConvertFromString(Text);
	//			Pending = true;
	//			SynchronisationHandler.SynchroniseSet(Registry, Key, num, val => Pending = false, e => Errored = true);
	//		}
	//		catch (Exception)
	//		{
	//			Errored = true;
	//		}
	//	}
	//}
	[ParameterVisualiser(typeof(float), Priority = VisualiserPriority.Lower)]
	[ParameterVisualiser(typeof(double), Priority = VisualiserPriority.Lower)]
	[ParameterVisualiser(typeof(long), Priority = VisualiserPriority.Lower)]
	[ParameterVisualiser(typeof(int), Priority = VisualiserPriority.Lower)]
	[ParameterVisualiser(typeof(short), Priority = VisualiserPriority.Lower)]
	public class DynamicConverterBox : SigmaTextBox
	{
		/// <summary>
		/// The converter that converts the given type for the registry
		/// </summary>
		public TypeConverter Converter { get; protected set; }

		public object CurrentValue { get; protected set; }

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			object obj = SynchronisationHandler.SynchroniseGet<object>(Registry, Key);

			if (Converter == null && obj != null)
			{
				Converter = TypeDescriptor.GetConverter(obj.GetType());
			}

			if (obj != null)
			{
				CurrentValue = obj;
				Text = obj.ToString();
			}
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			try
			{
				object convertedVal = Converter.ConvertFromString(Text);
				Pending = true;
				SynchronisationHandler.SynchroniseSet(Registry, Key, convertedVal, val => Pending = false, e => Errored = true);
			}
			catch (Exception)
			{
				Errored = true;

#if DEBUG
				//TODO: such an ugly hack only for testing
				Read();
#endif
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
