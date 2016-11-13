/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Sigma.Core.Data
{
	/// <summary>
	/// A default implementation of the databuffer interface.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	[Serializable]
	public class DataBuffer<T> : IDataBuffer<T>
	{
		[NonSerialized]
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private ILargeChunkedArray<T> data;
		private long length;
		private long offset;
		private long relativeOffset;

		[NonSerialized]
		private IDataBuffer<T> underlyingBuffer;
		[NonSerialized]
		private IDataBuffer<T> underlyingRootBuffer;

		public long Length
		{
			get { return length; }
		}

		public long Offset
		{
			get { return offset; }
		}

		public long RelativeOffset
		{
			get { return relativeOffset; }
		}

		public IDataType Type
		{
			get; private set;
		}

		public ILargeChunkedArray<T> Data
		{
			get { return data; }
		}

		/// <summary>
		/// Create a data buffer of a certain type with a certain underlying buffer.
		/// </summary>
		/// <param name="underlyingBuffer">The underlying buffer.</param>
		/// <param name="offset">The offset relative to the underlying buffer.</param>
		/// <param name="length">The length this buffer should have.</param>
		public DataBuffer(IDataBuffer<T> underlyingBuffer, long offset, long length)
		{
			if (underlyingBuffer == null)
			{
				throw new ArgumentNullException("Underlying buffer cannot be null.");
			}

			CheckBufferBounds(offset, length, offset + length, underlyingBuffer.Length);

			this.length = length;
			this.relativeOffset = offset;
			this.offset = offset + underlyingBuffer.Offset;

			this.data = underlyingBuffer.Data;
			this.Type = underlyingBuffer.Type;
			this.underlyingBuffer = underlyingBuffer;
			this.underlyingRootBuffer = underlyingBuffer.GetUnderlyingRootBuffer() == null ? underlyingBuffer : underlyingBuffer.GetUnderlyingRootBuffer();
		}

		/// <summary>
		/// Create a data buffer of a certain large chunked array.
		/// </summary>
		/// <param name="data">The large chunked array data.</param>
		/// <param name="underlyingType">The underlying data type (inferred if not given explicitly).</param>
		public DataBuffer(LargeChunkedArray<T> data, IDataType underlyingType = null) : this(data, 0L, data != null ? data.Length : 0L, underlyingType)
		{
		}

		/// <summary>
		/// Create a data buffer of a certain large chunked array.
		/// </summary>
		/// <param name="data">The large chunked array data.</param>
		/// <param name="underlyingType">The underlying data type (inferred if not given explicitly).</param>
		/// <param name="offset">The offset relative to the data array.</param>
		/// <param name="length">The length this buffer should have.</param>
		public DataBuffer(LargeChunkedArray<T> data, long offset, long length, IDataType underlyingType = null)
		{
			if (data == null)
			{
				throw new ArgumentNullException("Data cannot be null.");
			}

			CheckBufferBounds(offset, length, offset + length, data.Length);

			this.data = data;
			this.length = length;
			this.relativeOffset = offset;
			this.offset = offset;

			this.Type = InferDataType(underlyingType);
		}

		/// <summary>
		/// Create a data buffer of a certain array.
		/// </summary>
		/// <param name="data">The data array.</param>
		/// <param name="underlyingType">The underlying type (inferred if not explicitly given).</param>
		public DataBuffer(T[] data, IDataType underlyingType = null) : this(data, 0L, data != null ? data.Length : 0L, underlyingType)
		{
		}

		/// <summary>
		/// Create a data buffer of a certain array within certain bounds.
		/// </summary>
		/// <param name="data">The data array.</param>
		/// <param name="offset">The offset (relative to the given data array).</param>
		/// <param name="length">The length this buffer should have.</param>
		/// <param name="underlyingType">The underlying type (inferred if not explicitly given).</param>
		public DataBuffer(T[] data, long offset, long length, IDataType underlyingType = null)
		{
			if (data == null)
			{
				throw new ArgumentNullException("Data cannot be null.");
			}

			CheckBufferBounds(offset, length, offset + length, data.Length);

			this.data = new LargeChunkedArray<T>(data);
			this.length = length;
			this.relativeOffset = offset;
			this.offset = offset;

			this.Type = InferDataType(underlyingType);
		}

		/// <summary>
		/// Create a data buffer of a certain length.
		/// </summary>
		/// <param name="length">The length this buffer should have.</param>
		/// <param name="underlyingType">The underlying type (inferred if not explicitly given).</param>
		public DataBuffer(long length, IDataType underlyingType = null)
		{
			if (length < 1)
			{
				throw new ArgumentException($"Length must be >= 1 but was {length}.");
			}

			this.length = length;
			this.data = new LargeChunkedArray<T>(length);

			this.Type = InferDataType(underlyingType);
		}

		/// <summary>
		/// Copy constructor.
		/// </summary>
		/// <param name="other">The buffer to copy.</param>
		public DataBuffer(DataBuffer<T> other)
		{
			this.underlyingBuffer = other.underlyingBuffer;
			this.underlyingRootBuffer = other.underlyingRootBuffer;
			this.Type = other.Type;
			this.data = other.data;
			this.offset = other.offset;
			this.relativeOffset = other.relativeOffset;
			this.length = other.length;
		}

		private void CheckBufferBounds(long offset, long length, long requestedEndPosition, long underlyingLength)
		{
			if (offset < 0)
			{
				throw new ArgumentException($"Offset must be > 0 but was {offset}.");
			}

			if (length < 1)
			{
				throw new ArgumentException($"Length must be >= 1 but was {length}.");
			}

			if (requestedEndPosition > underlyingLength)
			{
				throw new ArgumentException($"Buffer length (+offset) cannot exceed length of its underlying buffer, but underlying buffer length was {underlyingLength} and requested length + offset {requestedEndPosition}.");
			}
		}

		private IDataType InferDataType(IDataType givenType)
		{
			if (givenType != null)
			{
				return givenType;
			}

			try
			{
				return DataTypes.GetMatchingType(typeof(T));
			}
			catch (ArgumentException e)
			{
				throw new ArgumentException($"Could not infer type interface for underlying system type {typeof(T)} (system type not registered) and no data type interface was explicitly given.", e);
			}
		}

		public IDataBuffer<T> Copy()
		{
			return new DataBuffer<T>(this);
		}

		public T GetValue(long index)
		{
			return data.GetValue(offset + index);
		}

		public TOther GetValueAs<TOther>(long index)
		{
			return (TOther) Convert.ChangeType(data.GetValue(offset + index), typeof(TOther));
		}

		public IDataBuffer<T> GetValues(long startIndex, long length)
		{
			return new DataBuffer<T>(this, startIndex, length);
		}

		public IDataBuffer<TOther> GetValuesAs<TOther>(long startIndex, long length)
		{
			LargeChunkedArray<TOther> otherData = new LargeChunkedArray<TOther>(length);

			otherData.FillWith<T>(this.data, this.offset + startIndex, 0L, length);

			return new DataBuffer<TOther>(otherData, 0L, length);
		}

		public ILargeChunkedArray<T> GetValuesArray(long startIndex, long length)
		{
			LargeChunkedArray<T> valuesArray = new LargeChunkedArray<T>(length);

			valuesArray.FillWith(this.data, this.offset + startIndex, 0L, length);

			return valuesArray;
		}

		public ILargeChunkedArray<TOther> GetValuesArrayAs<TOther>(long startIndex, long length)
		{
			LargeChunkedArray<TOther> valuesArray = new LargeChunkedArray<TOther>(length);

			valuesArray.FillWith<T>(this.data, this.offset + startIndex, 0L, length);

			return valuesArray;
		}

		public void SetValue(T value, long index)
		{
			data.SetValue(value, index + this.offset);
		}

		public void SetValues(IDataBuffer<T> buffer, long sourceStartIndex, long destStartIndex, long length)
		{
			this.data.FillWith(buffer.Data, sourceStartIndex + buffer.Offset, destStartIndex + this.offset, length);
		}

		public void SetValues(T[] values, long sourceStartIndex, long destStartIndex, long length)
		{
			this.data.FillWith(values, sourceStartIndex, destStartIndex + this.offset, length);
		}

		public IDataBuffer<T> GetUnderlyingBuffer()
		{
			return underlyingBuffer;
		}

		public IDataBuffer<T> GetUnderlyingRootBuffer()
		{
			return underlyingRootBuffer;
		}

		public IEnumerator<T> GetEnumerator()
		{
			for (long i = 0; i < this.length; i++)
			{
				yield return data.GetValue(i);
			}
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			for (long i = 0; i < this.length; i++)
			{
				yield return data.GetValue(i);
			}
		}
	}
}
