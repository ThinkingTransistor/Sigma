using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data
{
	/// <summary>
	/// A data buffer view with data of a certain type. Data buffers can be "stacked" hierarchically, meaning data buffers can represent a buffer view of their underlying data.
	/// </summary>
	/// <typeparam name="T">The underlying data type.</typeparam>
	public interface IDataBuffer<T> : IEnumerable<T>
	{
		/// <summary>
		/// The number of elements in this data buffer.
		/// </summary>
		long Length { get; }

		/// <summary>
		/// The absolute offset to the underlying root data buffer. 
		/// </summary>
		long Offset { get; }

		/// <summary>
		/// The underlying data type to use (e.g. FLOAT32).
		/// </summary>
		IDataType Type { get; }

		ILargeChunkedArray<T> Data { get; }

		/// <summary>
		/// Get the value at a certain index.
		/// </summary>
		/// <param name="index">The index.</param>
		/// <returns>The value at the given index.</returns>
		T GetValue(long index);

		/// <summary>
		/// Get the value at a certain index and attempt to explicitly cast to the given type. 
		/// </summary>
		/// <typeparam name="TOther">The type to cast to.</typeparam>
		/// <param name="index">The index.</param>
		/// <returns>The value at the given index, cast to given type.</returns>
		TOther GetValueAs<TOther>(long index);

		/// <summary>
		/// Get a new data buffer view with values referring to this buffer within a certain range. 
		/// </summary>
		/// <param name="startIndex">The start index referring to this buffer (not underlying).</param>
		/// <param name="length">The length.</param>
		/// <returns></returns>
		IDataBuffer<T> GetValues(long startIndex, long length);

		/// <summary>
		/// Get a new data buffer with the given type, with an underlying COPY of all data within the specified range, and attempt to cast all data to the new type. 
		/// </summary>
		/// <param name="startIndex">The start index referring to this buffer (not underlying).</param>
		/// <param name="length">The length.</param>
		/// <returns>A new data buffer of the given type with a COPY of the specified range.</returns>
		IDataBuffer<TOther> GetValuesAs<TOther>(long startIndex, long length);

		/// <summary>
		/// Set the value at the given index.
		/// </summary>
		/// <param name="value">The value.</param>
		/// <param name="index">The index.</param>
		void SetValue(T value, long index);

		/// <summary>
		/// Set values of this data buffer to the given values within the specified bounds.
		/// </summary>
		/// <param name="values">The values.</param>
		/// <param name="sourceStartIndex">Where to start copying in the source array (values).</param>
		/// <param name="destStartIndex">Where to paste within the destination, RELATIVE to the destination buffer (this data buffer).</param>
		/// <param name="length">Starting at the source start index, how many elements to copy.</param>
		void SetValues(T[] values, long sourceStartIndex, long destStartIndex, long length);

		/// <summary>
		/// Set values of this data buffer to the given values within the specified bounds.
		/// </summary>
		/// <param name="buffer">The data buffer to copy the data from.</param>
		/// <param name="sourceStartIndex">Where to start copying in the source buffer, RELATIVE to the source buffer.</param>
		/// <param name="destStartIndex">Where to paste within the destination, RELATIVE to the destination buffer (this data buffer).</param>
		/// <param name="destLength">Starting at the source start index, how many elements to copy.</param>
		void SetValues(IDataBuffer<T> buffer, long sourceStartIndex, long destStartIndex, long length);

		/// <summary>
		/// Copies this the data buffer, keeping the same underlying data (underlying buffers are not copied).
		/// </summary>
		/// <returns>A new data buffer view of the same data.</returns>
		IDataBuffer<T> Copy();

		/// <summary>
		/// Get the underlying data buffer.
		/// </summary>
		/// <returns>The underlying data buffer.</returns>
		IDataBuffer<T> GetUnderlyingBuffer();

		/// <summary>
		/// Get the underlying root data buffer (the buffer at the very top, where the actual data resides).
		/// </summary>
		/// <returns>The underlying root data buffer.</returns>
		IDataBuffer<T> GetUnderlyingRootBuffer();
	}
}