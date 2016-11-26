/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Control.SigmaGrid;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using WPFGrid = System.Windows.Controls.Grid;
// ReSharper disable InconsistentNaming

namespace Sigma.Core.Monitors.WPF.View.Tabs
{
	/// <summary>
	/// This class is basically only a wrapper for the tabs to be handled easily (and future proof)
	/// </summary>
	public class TabUI : UIWrapper<TabItem>
	{
		/// <summary>
		/// The <see cref="GridSize"/> of the tab.
		/// </summary>
		private GridSize _gridSize;

		/// <summary>
		/// The <see cref="GridSize"/> of the tab. 
		/// This value can only be changed if no grid has been created.
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
		/// The grid that holds the actual content.
		/// Use this value with caution - only use it to layout
		/// it in code. 
		/// </summary>
		public GridLayout Grid { get; private set; }

		public TabUI(string header, GridSize gridsize)
		{
			Content.Header = header;
			GridSize = gridsize;
		}

		/// <summary>
		/// Check whether a position of a grid is empty. It does not check if the position would be
		/// off grid and may throw an exception.
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
		/// Check whether a range on a grid is empty. It also checks if the position would be
		/// off grid.
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
			if (row + rowSpan > GridSize.Rows || column + columnSpan > GridSize.Columns) return false;

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

		private bool FindEmptyArea(WPFGrid grid, int rowSpan, int columnSpan, out int rowPosition, out int columnPosition)
		{
			for (int row = 0; row < GridSize.Rows; row++)
			{
				for (int column = 0; column < GridSize.Columns; column++)
				{
					//We know that it won't be out of range
					//So we check it fast
					if (!IsEmptyFast(grid, row, column)) continue;

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

		public void AddCumulativeElement(UIElement element, int rowSpan = 1, int columnSpan = 1)
		{
			if (rowSpan <= 0) throw new ArgumentOutOfRangeException(nameof(rowSpan));
			if (columnSpan <= 0) throw new ArgumentOutOfRangeException(nameof(columnSpan));

			EnsureGrid();

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
		/// Add a card onto the grid. 
		/// If (for some reasons) a <see cref="UIElement"/> has to be placed use <see cref="AddElement"/> instead.
		/// </summary>
		/// <param name="card">The card that will be added.</param>
		/// <param name="row">The row in which the card should be added.</param>
		/// <param name="column">The column in which the card should be added.</param>
		/// <param name="rowSpan">How many rows the card uses.</param>
		/// <param name="columnSpan">How many columns the card uses.</param>
		public void AddCard(Card card, int row, int column, int rowSpan = 1, int columnSpan = 1)
		{
			AddElement(card, row, column, rowSpan, columnSpan);

			//TODO: change in style? 
			card.HorizontalAlignment = HorizontalAlignment.Stretch;
			card.VerticalAlignment = VerticalAlignment.Stretch;
		}

		/// <summary>
		/// Place an element onto the grid. Normally a <see cref="Card"/> should be added
		/// for a consistent UI look-
		/// </summary>
		/// <param name="element">The element that will be added.</param>
		/// <param name="row">The row in which the element should be added.</param>
		/// <param name="column">The column in which the element should be added.</param>
		/// <param name="rowSpan">How many rows the element uses.</param>
		/// <param name="columnSpan">How many columns the element uses.</param>
		public void AddElement(UIElement element, int row, int column, int rowSpan = 1, int columnSpan = 1)
		{
			if (rowSpan <= 0) throw new ArgumentOutOfRangeException(nameof(rowSpan));
			if (columnSpan <= 0) throw new ArgumentOutOfRangeException(nameof(columnSpan));

			EnsureGrid();

			if (row + rowSpan > GridSize.Rows) throw new ArgumentException("Element would be out of range! (Too few rows)");
			if (column + columnSpan > GridSize.Columns) throw new ArgumentException("Element would be out of range! (Too few columns)");

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
		}

		/// <summary>
		/// This method ensures that the grid has been created and initialised. 
		/// </summary>
		private void EnsureGrid()
		{
			if (Grid == null)
			{
				Grid = CreateGrid();
				Content.Content = Grid;
			}
		}

		/// <summary>
		/// Create the grid with specified rows and columns. 
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

			return grid;
		}
	}
}
