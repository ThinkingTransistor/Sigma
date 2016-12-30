/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.Model.UI.Windows
{
	/// <summary>
	///     Defines the grid size for every Tab.
	/// </summary>
	public class GridSize : IDeepCopyable
	{
		/// <summary>
		///     Count of columns.
		/// </summary>
		private int _columns;

		/// <summary>
		///     Count of rows.
		/// </summary>
		private int _rows;

		/// <summary>
		///     Decide whether the <see cref="GridSize" /> is sealed.
		///     Once sealed, it should not be unsealed. (Cannot be unsealed by
		///     user)
		/// </summary>
		private bool _sealed;

		/// <summary>
		///     Create a new <see cref="GridSize" /> based on the parameters.
		/// </summary>
		/// <param name="rows">The row count (must be bigger than zero).</param>
		/// <param name="columns">The column count (must be bigger than zero).</param>
		public GridSize(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		/// <summary>
		///     Create a new <see cref="GridSize" /> based on the parameters.
		/// </summary>
		/// <param name="rows">The row count (must be bigger than zero).</param>
		/// <param name="columns">The column count (must be bigger than zero).</param>
		/// <param name="seal">
		///     If <code>true</code> seals the <see cref="GridSize" />. See <see cref="Sealed" /> for additional
		///     details on sealing.
		/// </param>
		public GridSize(int rows, int columns, bool seal) : this(rows, columns)
		{
			_sealed = seal;
		}

		/// <summary>
		///     Decide whether the <see cref="GridSize" /> is sealed.
		///     Once sealed, it cannot be unsealed.
		/// </summary>
		public bool Sealed
		{
			get { return _sealed; }
			set
			{
				if (!_sealed)
					_sealed = value;
				//If they try to unseal...
				else if (value == false)
					throw new ArgumentException($"Already sealed {nameof(GridSize)} - create a new {nameof(GridSize)}");
			}
		}

		/// <summary>
		///     Property for the row count. Row count must be bigger than zero.
		/// </summary>
		public int Rows
		{
			get { return _rows; }
			set
			{
				if (value <= 0)
					throw new ArgumentException("Rows may not be smaller or equal to zero.");
				if (_sealed)
					throw new ArgumentException($"{nameof(Rows)} already sealed");

				_rows = value;
			}
		}

		/// <summary>
		///     Property for the column count. Column count must be bigger than zero.
		/// </summary>
		public int Columns
		{
			get { return _columns; }
			set
			{
				if (value <= 0)
					throw new ArgumentException("Columns may not be smaller or equal to zero.");
				if (_sealed)
					throw new ArgumentException($"{nameof(Columns)} already sealed");

				_columns = value;
			}
		}

		public object DeepCopy()
		{
			return new GridSize(_rows, _columns, _sealed);
		}

		/// <summary>
		///     Set the dimensions of the <see cref="GridSize" />.
		/// </summary>
		/// <param name="rows">The row count (must be bigger than zero).</param>
		/// <param name="columns">The column count (must be bigger than zero).</param>
		public void Set(int rows, int columns)
		{
			Rows = rows;
			Columns = columns;
		}

		/// <summary>
		///     Check whether the passed parameters are suitable to create a new <see cref="GridSize" />.
		/// </summary>
		/// <param name="arr">The passed dimensions for the <see cref="GridSize" />. </param>
		private static void CheckDimensions(int[] arr)
		{
			if (arr == null) throw new ArgumentNullException(nameof(arr));

			if (arr.Length != 2)
				throw new ArgumentException("Only one dimensional arrays with two elements supported {rows, columns}!");
		}

		/// <summary>
		///     Convert a given Array to a <see cref="GridSize" />.
		/// </summary>
		/// <param name="arr">The passed <code>int[]</code> has to contain two parameters which both are greater than zero.</param>
		public static implicit operator GridSize(int[] arr)
		{
			CheckDimensions(arr);

			return new GridSize(arr[0], arr[1]);
		}

		/// <summary>
		///     Implicitly convert given <see cref="GridSize" /> to a <code>int[]</code>.
		/// </summary>
		/// <param name="grid">The <see cref="GridSize" /> subject of conversion.</param>
		public static explicit operator int[](GridSize grid)
		{
			return new[] {grid.Rows, grid.Columns};
		}

		public override string ToString()
		{
			return $"{Rows}, {Columns}";
		}

		public static bool operator ==(GridSize a, GridSize b)
		{
			if (a == null) throw new ArgumentNullException(nameof(a));
			if (b == null) throw new ArgumentNullException(nameof(b));

			return a.Equals(b);
		}

		public static bool operator !=(GridSize a, GridSize b)
		{
			return !(a == b);
		}

		protected bool Equals(GridSize other)
		{
			return (_rows == other._rows) && (_columns == other._columns);
		}

		public override bool Equals(object obj)
		{
			if (ReferenceEquals(null, obj)) return false;
			if (ReferenceEquals(this, obj)) return true;
			return (obj.GetType() == GetType()) && Equals((GridSize) obj);
		}

		public override int GetHashCode()
		{
			unchecked
			{
				return (_rows*397) ^ _columns;
			}
		}
	}
}