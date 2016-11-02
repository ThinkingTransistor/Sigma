/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data
{
	/// <summary>
	/// A (typically large) chunked array of any type. Behaves like an array but the data is actually split into chunks in a two-dimensional array. Typically used when the normal, one-dimensional array limit (2^32 / 2) is not enough or inconvenient. 
	/// </summary>
	/// <typeparam name="T"></typeparam>
	public interface ILargeChunkedArray<T>
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
	}

	/// <summary>
	/// A large, chunked and fast array to back the data buffer implementation. Supports very large arrays with up to 2PB (or 2048TB or 2097152GB) - hello future. 
	/// Realistically, you might run into system memory issues when getting anywhere near that size and it would be more efficient better to use record blocks.
	/// </summary>
	/// <typeparam name="T"></typeparam>
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
			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = data.GetValue(i + destStartIndex);
			}
		}

		public void FillWith<TOther>(ILargeChunkedArray<TOther> data, long sourceStartIndex, long destStartIndex, long length)
		{
			System.Type ownType = typeof(T);

			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = (T) Convert.ChangeType(data.GetValue(i + sourceStartIndex), ownType);
			}
		}

		public void FillWith(T[] data, long sourceStartIndex, long destStartIndex, long length)
		{
			for (long i = 0; i < length; i++)
			{
				this[i + destStartIndex] = data[i + sourceStartIndex];
			}
		}
	}
}
