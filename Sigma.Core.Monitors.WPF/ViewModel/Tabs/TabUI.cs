/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using log4net;
using Sigma.Core.Monitors.WPF.Model.SigmaGrid;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.Panels;
using Sigma.Core.Monitors.WPF.View;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls;
using WPFGrid = System.Windows.Controls.Grid;

// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.ViewModel.Tabs
{
	/// <summary>
	///     This class is basically only a wrapper for the tabs to be handled easily (and future proof if Dragablz is not supported anymore in near future).
	/// </summary>
	public class TabUI : UIWrapper<TabItem>
	{
		/// <summary>
		///     The log for <see cref="TabUI" />.
		/// </summary>
		private readonly ILog _log = LogManager.GetLogger(typeof(TabUI));

		/// <summary>
		///     The <see cref="GridSize" /> of the tab.
		/// </summary>
		private GridSize _gridSize;

		/// <summary>
		/// The monitor this tab is assigned to.
		/// </summary>
		protected WPFMonitor Monitor;

		/// <summary>
		///     Create a new <see cref="TabUI" /> - this basically is a
		///     <see cref="TabItem" /> with additional control.
		/// </summary>
		/// <param name="monitor">The monitor this tab is assigned to.</param>
		/// <param name="header">The header of the tab (name in the <see cref="TabControl" />)</param>
		/// <param name="gridsize">
		/// The <see cref="GridSize" />. Use
		/// <see cref="SigmaWindow.DefaultGridSize" />.
		/// </param>
		public TabUI(WPFMonitor monitor, string header, GridSize gridsize)
		{
			Monitor = monitor;
			Content.Header = header;
			GridSize = gridsize;
		}

		/// <summary>
		///     The <see cref="GridSize" /> of the tab.
		///     This value can only be changed if no grid has been created.
		/// </summary>
		public GridSize GridSize
		{
			get { return _gridSize; }
			set
			{
				if (Grid == null)
				{
					_gridSize = value;
				}
			}
		}

		/// <summary>
		///     The grid that contains the actual content.
		///     Use this value with caution - only for
		///     layout modification at runtime!
		/// </summary>
		public GridLayout Grid { get; private set; }

		/// <summary>
		///     Check whether a position of a grid is empty. It does not check if the position would be
		///     off grid and may throw an exception.
		/// </summary>
		/// <param name="grid">The grid, positions will be checked against.</param>
		/// <param name="row">The index of the row to check whether its occupied.</param>
		/// <param name="column">The index of the column to check whether its occupied.</param>
		/// <returns><code>false</code> when the position is occupied - <code>true</code> otherwise. </returns>
		private static bool IsEmptyFast(WPFGrid grid, int row, int column)
		{
			return !grid.GetChildAt(row, column).Any();
		}

		/// <summary>
		///     Check whether a range on a grid is empty. It also checks if the position would be
		///     off grid.
		/// </summary>
		/// <param name="grid">The grid, positions will be checked against.</param>
		/// <param name="row">The index of the start row to check whether its occupied.</param>
		/// <param name="column">The index of the start column to check whether its occupied.</param>
		/// <param name="rowSpan">How tall the row is.</param>
		/// <param name="columnSpan">How wide the column is.</param>
		/// <returns><code>false</code> when the range is occupied or the element is out of range - <code>true</code> otherwise. </returns>
		private bool IsEmpty(WPFGrid grid, int row, int column, int rowSpan, int columnSpan)
		{
			//If positions would be out of range
			if ((row + rowSpan > GridSize.Rows) || (column + columnSpan > GridSize.Columns))
			{
				return false;
			}

			//Check every element if its empty
			for (int i = row; i < row + rowSpan; i++)
			{
				for (int j = column; j < column + columnSpan; j++)
				{
					if (!IsEmptyFast(grid, i, j))
					{
						return false;
					}
				}
			}

			return true;
		}

		/// <summary>
		///     This method finds an empty area suitable for an object with given rowSpan and
		///     columnSpan.
		/// </summary>
		/// <param name="grid">The grid it will be checked against.</param>
		/// <param name="rowSpan">How many rows the new element requires.</param>
		/// <param name="columnSpan">How many columns the new element requires.</param>
		/// <param name="rowPosition">The index of the row where the element will fit. </param>
		/// <param name="columnPosition">The index of the column where the element will fit.</param>
		/// <returns><c>True</c> if there is enough room for the new element. <c>False</c> otherwise. </returns>
		private bool FindEmptyArea(WPFGrid grid, int rowSpan, int columnSpan, out int rowPosition, out int columnPosition)
		{
			for (int row = 0; row < GridSize.Rows; row++)
			{
				for (int column = 0; column < GridSize.Columns; column++)
				{
					//We know that it won't be out of range
					//So we check it fast
					if (!IsEmptyFast(grid, row, column))
					{
						continue;
					}

					if (IsEmpty(Grid, row, column, rowSpan, columnSpan))
					{
						rowPosition = row;
						columnPosition = column;
						return true;
					}
				}
			}

			rowPosition = -1;
			columnPosition = -1;

			return false;
		}

		/// <summary>
		///     Apply the legend colour-coding to a passed panel.
		/// </summary>
		/// <param name="panel">The given panel. </param>
		/// <param name="legend">The given panel. </param>
		protected virtual void ApplyLegend(SigmaPanel panel, StatusBarLegendInfo legend)
		{
			if (legend != null)
			{
				panel.Header.Background = new SolidColorBrush(legend.LegendColor);

				if (legend.ForegroundColor.HasValue)
				{
					SolidColorBrush brush = new SolidColorBrush(legend.ForegroundColor.Value);

					foreach (object headerChild in panel.Header.Children)
					{
						Control control = headerChild as Control;
						if (control != null)
						{
							control.Foreground = brush;
						}
					}
				}
			}
		}

		/// <summary>
		///     Add a <see cref="SigmaPanel" /> cumulatively to the tab. (At
		///     the next free position; may skips empty panels if columnSpan
		///     or rowSpan is too big.
		/// </summary>
		/// <param name="panel">The <see cref="SigmaPanel" /> that will be added.</param>
		/// <param name="rowSpan">How many rows the panel requires. </param>
		/// <param name="columnSpan">How many columns the panel requires. </param>
		/// <param name="legend">Mark a panel to a special colour / legend. </param>
		/// <exception cref="ArgumentOutOfRangeException">
		///     If rowSpan is smaller or equal to zero or if columnSpan
		///     is smaller or equal to zero.
		/// </exception>
		/// <exception cref="IndexOutOfRangeException">If there is no space for the new <see cref="SigmaPanel" />. </exception>
		public void AddCumulativePanel(SigmaPanel panel, int rowSpan = 1, int columnSpan = 1,
			StatusBarLegendInfo legend = null)
		{
			panel.Monitor = Monitor;
			AddCumulativeElement(panel, rowSpan, columnSpan);
			ApplyLegend(panel, legend);
			panel.Initialise(Monitor.Window);
		}

		/// <summary>
		///     Add a <see cref="UIElement" /> cumulatively to the tab. (At
		///     the next free position; may skips empty panels if columnSpan
		///     or rowSpan is too big.
		/// </summary>
		/// <param name="element">The <see cref="UIElement" /> that will be added.</param>
		/// <param name="rowSpan">How many rows the panel requires. </param>
		/// <param name="columnSpan">How many columns the panel requires. </param>
		/// <exception cref="ArgumentOutOfRangeException">
		///     If rowSpan is smaller or equal to zero or if columnSpan
		///     is smaller or equal to zero.
		/// </exception>
		/// <exception cref="IndexOutOfRangeException">If there is no space for the new <see cref="UIElement" />. </exception>
		public void AddCumulativeElement(UIElement element, int rowSpan = 1, int columnSpan = 1)
		{
			if (rowSpan <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(rowSpan));
			}
			if (columnSpan <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(columnSpan));
			}

			EnsureGridCreated();

			int row, column;

			if (FindEmptyArea(Grid, rowSpan, columnSpan, out row, out column))
			{
				AddElement(element, row, column, rowSpan, columnSpan);
			}
			else
			{
				throw new IndexOutOfRangeException("Grid is full or element too big!");
			}

		}

		/// <summary>
		///     Place an <see cref="SigmaPanel" /> onto the grid. For granular control use
		///     <see cref="AddElement" />.
		/// </summary>
		/// <param name="panel">The <see cref="SigmaPanel" /> that will be added.</param>
		/// <param name="row">The row in which the <see cref="SigmaPanel" /> should be added.</param>
		/// <param name="column">The column in which the <see cref="SigmaPanel" /> should be added.</param>
		/// <param name="rowSpan">How many rows the <see cref="SigmaPanel" /> uses.</param>
		/// <param name="columnSpan">How many columns the <see cref="SigmaPanel" /> uses.</param>
		/// <param name="legend">Mark a panel to a special colour / legend. </param>
		/// <exception cref="ArgumentOutOfRangeException">
		///     If rowSpan is smaller or equal to zero or if columnSpan
		///     is smaller or equal to zero.
		/// </exception>
		/// <exception cref="IndexOutOfRangeException">If there is no space for the new <see cref="SigmaPanel" />. </exception>
		public void AddPanel(SigmaPanel panel, int row, int column, int rowSpan = 1, int columnSpan = 1,
			StatusBarLegendInfo legend = null)
		{
			panel.Monitor = Monitor;
			AddElement(panel, row, column, rowSpan, columnSpan);
			ApplyLegend(panel, legend);
			panel.Initialise(Monitor.Window);
		}

		/// <summary>
		///     Place an <see cref="UIElement" /> onto the grid. Normally a <see cref="SigmaPanel" /> should be added
		///     to the UI for a consistent look and feel of the UI.
		/// </summary>
		/// <param name="element">The <see cref="UIElement" /> that will be added.</param>
		/// <param name="row">The row in which the <see cref="UIElement" /> should be added.</param>
		/// <param name="column">The column in which the <see cref="UIElement" /> should be added.</param>
		/// <param name="rowSpan">How many rows the <see cref="UIElement" /> uses.</param>
		/// <param name="columnSpan">How many columns the <see cref="UIElement" /> uses.</param>
		/// <exception cref="ArgumentOutOfRangeException">
		///     If rowSpan is smaller or equal to zero or if columnSpan
		///     is smaller or equal to zero.
		/// </exception>
		/// <exception cref="IndexOutOfRangeException">If there is no space for the new <see cref="UIElement" />. </exception>
		public void AddElement(UIElement element, int row, int column, int rowSpan = 1, int columnSpan = 1)
		{
			if (rowSpan <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(rowSpan));
			}
			if (columnSpan <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(columnSpan));
			}

			EnsureGridCreated();

			if (row + rowSpan > GridSize.Rows)
			{
				throw new IndexOutOfRangeException("Element would be out of range! (Too few rows)");
			}
			if (column + columnSpan > GridSize.Columns)
			{
				throw new IndexOutOfRangeException("Element would be out of range! (Too few columns)");
			}

			Grid.Children.Add(element);
			WPFGrid.SetRow(element, row);
			WPFGrid.SetColumn(element, column);
			WPFGrid.SetRowSpan(element, rowSpan);
			WPFGrid.SetColumnSpan(element, columnSpan);

			//TODO: Ugly hack - otherwise it does not work if AddElement is called after Prepare
			if (WrappedContent.IsSelected)
			{
				WrappedContent.IsSelected = false;
				WrappedContent.IsSelected = true;
			}

			_log.Debug($"Added {element.GetType().Name} at {row}, {column}, with a span of {rowSpan}, {columnSpan}");
		}

		/// <summary>
		///     This method ensures that the grid has been created and initialised.
		/// </summary>
		private void EnsureGridCreated()
		{
			if (Grid == null)
			{
				Grid = CreateGrid();
				Content.Content = Grid;
			}
		}

		/// <summary>
		///     Create the grid with specified rows and columns.
		/// </summary>
		/// <returns>Returns the newly created grid. </returns>
		private GridLayout CreateGrid()
		{
			GridLayout grid = new GridLayout();

			_gridSize = (GridSize) _gridSize.DeepCopy();
			_gridSize.Sealed = true;

			int[] gridSize = (int[]) _gridSize;

			//add rows and columns
			int rows = gridSize[0];
			for (int i = 0; i < rows; i++)
			{
				grid.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
			}

			int columns = gridSize[1];
			for (int i = 0; i < columns; i++)
			{
				grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
			}

			//TODO: change in style?
			grid.ChildMargin = new Thickness(10);

			_log.Debug($"The {nameof(GridLayout)} has been created with the size {gridSize[0]}, {gridSize[1]}. From now on it cannot be modified.");

			return grid;
		}
	}
}