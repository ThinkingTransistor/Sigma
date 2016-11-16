/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Sigma.Core.Data
{
	/// <summary>
	/// A (typically large) chunked array of any type. Behaves like an array but the data is actually split into chunks in a two-dimensional array. Typically used when the normal, one-dimensional array limit (2^32 / 2) is not enough or inconvenient. 
	/// </summary>
	/// <typeparam name="T"></typeparam>
	public interface ILargeChunkedArray<T> : IEnumerable<T>, IDeepCopyable
	{
		/// <summary>
		/// The chunked data. First dimension represents chunk index and second data index within chunk. 
		/// Note: The last array should be clipped in size to the actual size for easier out of bounds check. 
		/// </summary>
		T[][] ChunkedData { get; }

		/// <summary>
		/// The total length of this array.
		/// </summary>
		long Length { get; }

		/// <summary>
		/// Get a value at a certain index.
		/// </summary>
		/// <param name="index">The value at the given index.</param>
		/// <returns></returns>
		T GetValue(long index);

		/// <summary>
		/// Sets a given value at a certain index.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <param name="index">The index.</param>
		void SetValue(T value, long index);

		/// <summary>
		/// Get a COPY of all values within a certain range as a single array.
		/// </summary>
		/// <param name="startIndex">The start index.</param>
		/// <param name="length">The length.</param>
		/// <returns>An array with a COPY of all values within a certain range.</returns>
		T[] GetValuesPackedArray(long startIndex, int length);

		/// <summary>
		/// Attempt to get a COPY of all values starting at a certain index if they would fit inside a single one-dimensional (packed) array.
		/// </summary>
		/// <param name="startIndex">The optional start index (default: 0).</param>
		/// <returns>An array with a COPY of all values within a certain range (if possible).</returns>
		T[] TryGetValuesPackedArray(long startIndex = 0);

		/// <summary>
		/// Get a COPY of all values within a range as a chunked array.
		/// </summary>
		/// <param name="startIndex">The start index.</param>
		/// <param name="length">The length.</param>
		/// <returns>A COPY of all values within the given range as a chunked array.</returns>
		ILargeChunkedArray<T> GetValuesChunkedArray(long startIndex, long length);

		/// <summary>
		/// Fill this chunked array with data from another chunked array of the same type within the given range. 
		/// </summary>
		/// <param name="data">Another chunked array to copy the data from.</param>
		/// <param name="sourceStartIndex">The source start index (where to start copying in the source data).</param>
		/// <param name="destStartIndex">The destination start index (where to start pasting in the destination data - this chunked array).</param>
		/// <param name="length">The length (how many elements to copy).</param>
		void FillWith(ILargeChunkedArray<T> data, long sourceStartIndex, long destStartIndex, long length);

		/// <summary>
		/// Fill this chunked array with data from another array of the same type within the given range. 
		/// </summary>
		/// <param name="data">Another chunked array to copy the data from.</param>
		/// <param name="sourceStartIndex">The source start index (where to start copying in the source data).</param>
		/// <param name="destStartIndex">The destination start index (where to start pasting in the destination data - this chunked array).</param>
		/// <param name="length">The length (how many elements to copy).</param>
		void FillWith(T[] data, long sourceStartIndex, long destStartIndex, long length);

		/// <summary>
		/// Fill this chunked array with data from another chunked array of another same type within the given range (data may have to be cast).
		/// </summary>
		/// <typeparam name="TOther">The type of the source data.</typeparam>
		/// <param name="data">Another chunked array to copy the data from.</param>
		/// <param name="sourceStartIndex">The source start index (where to start copying in the source data).</param>
		/// <param name="destStartIndex">The destination start index (where to start pasting in the destination data - this chunked array).</param>
		/// <param name="length">The length (how many elements to copy).</param>
		void FillWith<TOther>(ILargeChunkedArray<TOther> data, long sourceStartIndex, long destStartIndex, long length);
	}

	/// <summary>
	/// A large, chunked and fast array to back the data buffer implementation. Supports very large arrays with up to 2PB (or 2048TB or 2097152GB) - hello future. 
	/// Realistically, you might run into system memory issues when getting anywhere near that size and it would be more efficient better to use record blocks.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	[Serializable]
	public class LargeChunkedArray<T> : ILargeChunkedArray<T>
	{
		// These absolutely need to be constant in order for the C# optimiser to inline the get / set methods (hint: it's much faster). 
		// The block size is so large that for most use cases, it behaves (and performs) just like a regular array. 
		internal const int BLOCK_SIZE = 1048576;
		internal const int BLOCK_SIZE_MASK = BLOCK_SIZE - 1;
		internal const int BLOCK_SIZE_LOG2 = 20;

		private T[][] data;

		public T[][] ChunkedData { get { return data; } }

		public long Length { get; private set; }

		/// <summary>
		/// Create a large chunked array representation of a certain data array.
		/// </summary>
		/// <param name="data"></param>
		public LargeChunkedArray(T[] data)
		{
			this.data = new T[1][];

			this.data[0] = data;

			this.Length = data.Length;
		}

		internal LargeChunkedArray(T[][] data, long length)
		{
			this.data = data;
			this.Length = length;
		}

		/// <summary>
		/// Create a large chunked array representation of a certain data array.
		/// </summary>
		/// <param name="size">The size this large chunked array should have.</param>
		public LargeChunkedArray(long size)
		{
			int numBlocks = (int) (size / BLOCK_SIZE);
			bool differentLastArray = (numBlocks * BLOCK_SIZE) < size;

			if (differentLastArray)
			{
				numBlocks++;
			}

			data = new T[numBlocks][];

			for (int i = 0; i < numBlocks - 1; i++)
			{
				data[i] = new T[BLOCK_SIZE];
			}

			//Jag the last array if the total size doesn't magically line up with the block size
			data[numBlocks - 1] = new T[differentLastArray ? size % BLOCK_SIZE : BLOCK_SIZE];

			Length = size;
		}

		public object DeepCopy()
		{
			return new LargeChunkedArray<T>((T[][]) this.data.Clone(), this.Length);
		}

		public T this[long index]
		{
			get
			{
				int blockIndex = (int) (index >> BLOCK_SIZE_LOG2);
				int indexWithinBlock = (int) (index & BLOCK_SIZE_MASK);

				return data[blockIndex][indexWithinBlock];
			}
			set
			{
				int blockIndex = (int) (index >> BLOCK_SIZE_LOG2);
				int indexWithinBlock = (int) (index & BLOCK_SIZE_MASK);

				data[blockIndex][indexWithinBlock] = value;
			}
		}

		public T GetValue(long index)
		{
			int blockIndex = (int) (index >> BLOCK_SIZE_LOG2);
			int indexWithinBlock = (int) (index & BLOCK_SIZE_MASK);

			return data[blockIndex][indexWithinBlock];
		}

		public void SetValue(T value, long index)
		{
			int blockIndex = (int) (index >> BLOCK_SIZE_LOG2);
			int indexWithinBlock = (int) (index & BLOCK_SIZE_MASK);

			data[blockIndex][indexWithinBlock] = value;
		}

		public void FillWith(ILargeChunkedArray<T> data, long sourceStartIndex, long destStartIndex, long length)
		{
			CheckLength(length);
			CheckDestStartIndex(destStartIndex);

			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = data.GetValue(i + sourceStartIndex);
			}
		}

		public void FillWith<TOther>(ILargeChunkedArray<TOther> data, long sourceStartIndex, long destStartIndex, long length)
		{
			CheckLength(length);
			CheckDestStartIndex(destStartIndex);

			System.Type ownType = typeof(T);

			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = (T) Convert.ChangeType(data.GetValue(i + sourceStartIndex), ownType);
			}
		}

		public void FillWith(T[] data, long sourceStartIndex, long destStartIndex, long length)
		{
			CheckLength(length);
			CheckDestStartIndex(destStartIndex);

			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = data[i + sourceStartIndex];
			}
		}

		public T[] TryGetValuesPackedArray(long startIndex = 0)
		{
			if (Length - startIndex >= Int32.MaxValue / 2)
			{
				throw new ArgumentException($"Cannot pack all values of this array within a single one-dimensional array (too long with {Length - startIndex} elements).");
			}

			return GetValuesPackedArray(startIndex, (int) (Length - startIndex));
		}

		public T[] GetValuesPackedArray(long startIndex, int length)
		{
			CheckLength(length);

			T[] array = new T[length];

			for (long i = 0; i < length; i++)
			{
				array[i] = this[startIndex + i];
			}

			return array;
		}

		public ILargeChunkedArray<T> GetValuesChunkedArray(long startIndex, long length)
		{
			CheckLength(length);

			LargeChunkedArray<T> array = new LargeChunkedArray<T>(length);

			for (long i = 0; i < length; i++)
			{
				array[i] = this[i + startIndex];
			}

			return array;
		}

		private void CheckLength(long length)
		{
			if (length < 1)
			{
				throw new ArgumentException($"Length has to be >= 1 (but was {length}).");
			}
		}

		private void CheckDestStartIndex(long startIndex)
		{
			if (startIndex < 0)
			{
				throw new ArgumentException($"Destination start index has to be >= 0 (but was {startIndex}).");
			}
		}

		public IEnumerator<T> GetEnumerator()
		{
			long length = this.Length;
			for (long i = 0; i < length; i++)
			{
				yield return this[i];
			}
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			long length = this.Length;
			for (long i = 0; i < length; i++)
			{
				yield return this[i];
			}
		}
	}
}
