/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Model.SigmaGrid
{
	/// <summary>
	/// A static class that provides functionality to the <see cref="Grid"/> class.
	/// </summary>
	public static class GridExtensions
	{
		// ReSharper disable once InconsistentNaming
		/// <summary>
		/// Get a <see cref="FrameworkElement"/> at a specific point ont the grid. 
		/// If multiple items have been added to the same position or some are overlapping (e.g. column span etc.)
		/// <b>all</b> are returned.
		/// </summary>
		/// <param name="instance">The grid itself.</param>
		/// <param name="row">The row to get a specific item.</param>
		/// <param name="column">The column to get a specific item.</param>
		/// <returns>A <see cref="IEnumerable{T}"/> that contains all elements that are inside that grid.</returns>
		public static IEnumerable<FrameworkElement> GetChildAt(this Grid instance, int row, int column)
		{
			if (null == instance)
			{
				throw new ArgumentNullException(nameof(instance));
			}

			List<FrameworkElement> list = new List<FrameworkElement>();
			foreach (FrameworkElement fe in instance.Children)
			{
				int rowStart = Grid.GetRow(fe);
				int rowEnd = rowStart + Grid.GetRowSpan(fe) - 1;
				int columnStart = Grid.GetColumn(fe);
				int columnEnd = columnStart + Grid.GetColumnSpan(fe) - 1;

				if (row >= rowStart && row <= rowEnd && column >= columnStart && column <= columnEnd)
				{
					list.Add(fe);
				}
			}

			return list;
		}
	}
}