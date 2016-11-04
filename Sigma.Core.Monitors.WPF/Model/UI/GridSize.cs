/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics;

namespace Sigma.Core.Monitors.WPF.Model.UI
{
	public class GridSize
	{
		private int rows, columns;

		public int Rows
		{
			get
			{
				return rows;
			}
			set
			{
				if (value <= 0)
				{
					throw new ArgumentException("Rows may not be smaller or equal to zero.");
				}

				rows = value;
			}
		}

		public int Columns
		{
			get
			{
				return columns;
			}
			set
			{
				if (value <= 0)
				{
					throw new ArgumentException("Columns may not be smaller or equal to zero.");
				}

				columns = value;
			}
		}


		public GridSize() : this(3, 4) { }

		public GridSize(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		public void Set(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		private static void CheckDimensions(int[] arr)
		{
			if (arr.Length != 2)
			{
				throw new ArgumentException("Only array with two elements supported {rows, columns}!");
			}
		}

		public static implicit operator GridSize(int[] arr)
		{
			CheckDimensions(arr);

			return new GridSize(arr[0], arr[1]);
		}

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
