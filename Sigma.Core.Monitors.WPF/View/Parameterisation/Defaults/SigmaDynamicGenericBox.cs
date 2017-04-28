using System;
using System.ComponentModel;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// This is a generic text box that automatically uses the type of the registry in order to correctly display and parse to / from the value.
	/// 
	/// It is necessary for XAML or other cases where generics are not possible.
	/// </summary>
	public class SigmaDynamicGenericBox : SigmaTextBox
	{
		/// <summary>
		/// The converter that converts the given type for the registry
		/// </summary>
		public TypeConverter Converter { get; protected set; }

		/// <summary>
		/// The current value that is displayed
		/// </summary>
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
			}
		}
	}
}