using System;
using System.ComponentModel;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// A textbox that can convert from and to an arbitrary object with a type converter.
	/// </summary>
	/// <typeparam name="T">The type that is currently being represented.</typeparam>
	[GenericParameterVisualiser(typeof(double), Priority = VisualiserPriority.Lower)]
	[GenericParameterVisualiser(typeof(float), Priority = VisualiserPriority.Lower)]
	[GenericParameterVisualiser(typeof(short), Priority = VisualiserPriority.Lower)]
	[GenericParameterVisualiser(typeof(int), Priority = VisualiserPriority.Lower)]
	[GenericParameterVisualiser(typeof(long), Priority = VisualiserPriority.Lower)]
	public class SigmaGenericBox<T> : SigmaTextBox
	{
		/// <summary>
		/// The converter that converts the given type for the registry
		/// </summary>
		public TypeConverter Converter { get; protected set; }

		/// <summary>
		/// The current active value.
		/// </summary>
		protected T CurrentValue;

		/// <summary>
		/// Create a generic box and initialise the converter.
		/// </summary>
		public SigmaGenericBox()
		{
			Converter = TypeDescriptor.GetConverter(typeof(T));
		}

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			SynchronisationHandler.SynchroniseUpdate(Registry, Key, CurrentValue, val =>
			{
				CurrentValue = val;
				Text = CurrentValue.ToString();
			});
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			try
			{
				T convertedValue = (T) Converter.ConvertFromString(Text);
				Pending = true;
				SynchronisationHandler.SynchroniseSet(Registry, Key, convertedValue, val =>
				{
					Pending = false;
					Errored = false;
				}, e => Errored = true);

			}
			catch (Exception)
			{
				Errored = true;
				throw;
			}
		}
	}
}