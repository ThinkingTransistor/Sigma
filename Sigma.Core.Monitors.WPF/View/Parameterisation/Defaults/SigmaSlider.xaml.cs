/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Monitors.WPF.ViewModel.Parameterisation;
using Sigma.Core.Utils;
using System;
using System.Globalization;
using System.Windows;

namespace Sigma.Core.Monitors.WPF.View.Parameterisation.Defaults
{
	/// <summary>
	/// A custom slider that implements <see cref="IParameterVisualiser"/> and can visualise values from a registry
	/// (doubles and flaots in this case).
	/// </summary>
	public partial class SigmaSlider
	{
		private ISynchronisationHandler _synchronisationHandler;
		private IRegistry _registry;
		private string _key;
		private double _maximum;
		private double _minimum;

		/// <summary>
		/// Decides wether the scale is lograithmic or not.
		/// </summary>
		public bool IsLogarithmic { get; set; }


		/// <summary>
		/// The fully resolved key to access the synchandler.
		/// </summary>
		public override string Key
		{
			get { return _key; }
			set
			{
				_key = value;
				TextBox.Key = value;
			}
		}

		/// <summary>
		/// The registry for which the visualiser displays values. (e.g. operators registry)
		/// </summary>
		public override IRegistry Registry
		{
			get { return _registry; }
			set
			{
				_registry = value;
				TextBox.Registry = value;
			}
		}

		/// <summary>
		/// The SynchronisationHandler that is used to sync the parameter with the training process.
		/// </summary>
		public override ISynchronisationHandler SynchronisationHandler
		{
			get { return _synchronisationHandler; }
			set
			{
				_synchronisationHandler = value;
				TextBox.SynchronisationHandler = value;
			}
		}

		/// <summary>
		/// Determines whether the parameter is editable or not. 
		/// </summary>
		public override bool IsReadOnly { get; set; }

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
		/// The minimal value that the slider will return. This is not the min value of the slider!
		/// The slider is ranging from [0;1]
		/// </summary>
		public double Minimum
		{
			get { return _minimum; }
			set
			{
				_minimum = value; 
				UpdateParameters(Minimum, Maximum, out B, out C);
			}
		}

		/// <summary>
		/// The maximal value that the slider will return. This is not the max value of the slider!
		/// The slider is ranging from [0;1]
		/// </summary>
		public double Maximum
		{
			get { return _maximum; }
			set
			{
				_maximum = value;
				UpdateParameters(Minimum, Maximum, out B, out C);
			}
		}

		/// <summary>
		/// The parameters of the exponential function.
		/// 10^(b*x) + c
		/// </summary>
		protected double B, C;

		protected double Tolerance { get; set; } = 0.0001;

		/// <summary>
		/// Force the visualiser to update its value (i.e. display the value that is stored).
		/// </summary>
		public override void Read()
		{
			TextBox.Read();
		}

		/// <summary>
		/// Force the visualiser to store its value (i.e. write the value that is displayed to the registry).
		/// </summary>
		public override void Write()
		{
			TextBox.Write();
		}

		public SigmaSlider(double min, double max)
		{
			Minimum = min;
			Maximum = max;

			InitializeComponent();
		}

		protected void UpdateParameters(double min, double max, out double b, out double c)
		{
			b = Math.Log10(max - min + 1);
			c = min - 1;
		}

		protected double DoCalculation(double number)
		{
			if (IsLogarithmic)
			{
				if (number < Tolerance)
				{
					return Minimum;
				}

				return C + Math.Pow(10, B * number);
			}

			return (Maximum - Minimum) * number + Minimum;
		}

		private void Slider_OnValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
		{
			TextBox.Text = DoCalculation(e.NewValue).ToString(CultureInfo.CurrentCulture);
		}
	}
}
