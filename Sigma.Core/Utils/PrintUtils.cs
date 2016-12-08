/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of utility functions and constants for pretty printing various things to console (string formatting).
	/// </summary>
	public static class PrintUtils
	{
		/// <summary>
		/// A UTF-16 greyscale palette from min to max.
		/// </summary>
		public static readonly char[] Utf16GreyscalePalette = {' ', '·', '-', '▴', '▪', '●', '♦', '■', '█'};

		/// <summary>
		/// An ASCII greyscale palette from min to max.
		/// </summary>
		public static readonly char[] AsciiGreyscalePalette = { ' ', '.', ':', 'x', 'T', 'Y', 'V', 'X', 'H', 'N', 'M' };
	}
}
