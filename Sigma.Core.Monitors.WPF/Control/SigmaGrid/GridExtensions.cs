using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;

namespace Sigma.Core.Monitors.WPF.Control.SigmaGrid
{
	public static class GridExtensions
	{
		// ReSharper disable once InconsistentNaming
		public static IEnumerable<FrameworkElement> GetChildAt(this Grid instance, int row, int column)
		{
			if (null == instance) throw new ArgumentNullException(nameof(instance));

			List<FrameworkElement> list = new List<FrameworkElement>();
			foreach (FrameworkElement fe in instance.Children)
			{
				int rowStart = Grid.GetRow(fe);
				int rowEnd = rowStart + Grid.GetRowSpan(fe) - 1;
				int columnStart = Grid.GetColumn(fe);
				int columnEnd = columnStart + Grid.GetColumnSpan(fe) - 1;

				if (row >= rowStart && row <= rowEnd && (column >= columnStart && column <= columnEnd))
				{
					list.Add(fe);
				}
			}

			return list;
		}
	}
}
