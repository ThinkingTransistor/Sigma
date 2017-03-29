/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Windows.Data;

namespace Sigma.Core.Monitors.WPF.Utils.Converters
{
	/// <summary>
	/// This converter inverses a given boolean.
	/// 
	/// <c>True</c> will become <c>False</c> and vice versa. 
	/// </summary>
	[ValueConversion(typeof(bool), typeof(bool))]
	public class InverseBooleanConverter : IValueConverter
	{
		#region IValueConverter Members

		/// <inheritdoc />
		public object Convert(object value, Type targetType, object parameter,
			System.Globalization.CultureInfo culture)
		{
			if (targetType != typeof(bool))
				throw new InvalidOperationException("The target must be a boolean");

			return value != null && !(bool)value;
		}

		/// <inheritdoc />
		public object ConvertBack(object value, Type targetType, object parameter,
			System.Globalization.CultureInfo culture)
		{
			throw new NotSupportedException();
		}

		#endregion
	}
}