/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Monitors.WPF.Model.UI.Windows
{
	/// <summary>
	/// Defines the grid size for every Tab.
	/// </summary>
	public class GridSize
	{
		/// <summary>
		/// Count of rows.
		/// </summary>
		private int _rows;
		/// <summary>
		/// Count of columns.
		/// </summary>
		private int _columns;

		/// <summary>
		/// Property for the row count. Row count must be bigger than zero.
		/// </summary>
		public int Rows
		{
			get
			{
				return _rows;
			}
			set
			{
				if (value <= 0)
				{
					throw new ArgumentException("Rows may not be smaller or equal to zero.");
				}

				_rows = value;
			}
		}

		/// <summary>
		/// Property for the column count. Column count must be bigger than zero.
		/// </summary>
		public int Columns
		{
			get
			{
				return _columns;
			}
			set
			{
				if (value <= 0)
				{
					throw new ArgumentException("Columns may not be smaller or equal to zero.");
				}

				_columns = value;
			}
		}

		/// <summary>
		/// Create a new <see cref="GridSize"/> based on the parameters.
		/// </summary>
		/// <param name="rows">The row count (must be bigger than zero).</param>
		/// <param name="columns">The column count (must be bigger than zero).</param>
		public GridSize(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		/// <summary>
		/// Set the dimensions of the <see cref="GridSize"/>.
		/// </summary>
		/// <param name="rows">The row count (must be bigger than zero).</param>
		/// <param name="columns">The column count (must be bigger than zero).</param>
		public void Set(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		/// <summary>
		/// Check whether the passed parameters are suitable to create a new <see cref="GridSize"/>.
		/// </summary>
		/// <param name="arr">The passed dimensions for the <see cref="GridSize"/>. </param>
		private static void CheckDimensions(int[] arr)
		{
			if (arr.Length != 2)
			{
				throw new ArgumentException("Only array with two elements supported {rows, columns}!");
			}
		}

		/// <summary>
		/// Convert a given Array to a <see cref="GridSize"/>.
		/// </summary>
		/// <param name="arr">The passed <see cref="int[]"/> has to contain two parameters which both are greater than zero.</param>
		public static implicit operator GridSize(int[] arr)
		{
			CheckDimensions(arr);

			return new GridSize(arr[0], arr[1]);
		}

		/// <summary>
		/// Implicitly convert given <see cref="GridSize"/> to a <see cref="int[]"/>.
		/// </summary>
		/// <param name="grid">The <see cref="GridSize"/> subject of conversion.</param>
		public static implicit operator int[] (GridSize grid)
		{
			return new int[] { grid.Rows, grid.Columns };
		}

		public override string ToString()
		{
			return $"{Rows}, {Columns}";
		}
	}
}
