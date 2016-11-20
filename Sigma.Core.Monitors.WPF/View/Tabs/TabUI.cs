/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Windows;
using System.Windows.Controls;
using MahApps.Metro.Behaviours;
using MaterialDesignThemes.Wpf;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
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
		public Grid Grid { get; private set; }

		public TabUI(string header, GridSize gridsize)
		{
			Content.Header = header;
			GridSize = gridsize;
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
			if (Grid == null)
			{
				Grid = CreateGrid();
				Content.Content = Grid;
			}

			Grid.Children.Add(element);
			Grid.SetRow(element, row);
			Grid.SetColumn(element, column);
			Grid.SetRowSpan(element, rowSpan);
			Grid.SetColumnSpan(element, columnSpan);

			//TODO: Ugly hack - otherwise it does not work if AddElement is called after Prepare
			if (WrappedContent.IsSelected)
			{
				WrappedContent.IsSelected = false;
				WrappedContent.IsSelected = true;
			}
		}

		/// <summary>
		/// Create the grid with specified rows and columns. 
		/// </summary>
		/// <returns>Returns the newly created grid. </returns>
		private Grid CreateGrid()
		{
			Grid grid = new Grid();

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
				grid.ColumnDefinitions.Add(new ColumnDefinition() { Width = new GridLength(1, GridUnitType.Star) });
			}

			//TODO: change in style?
			grid.Margin = new Thickness(20);
			//TODO: padding

			return grid;
		}
	}
}
