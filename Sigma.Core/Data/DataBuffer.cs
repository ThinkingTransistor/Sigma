/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Sigma.Core.Utils;

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
		private readonly IDataBuffer<T> _underlyingBuffer;
		[NonSerialized]
		private readonly IDataBuffer<T> _underlyingRootBuffer;

		public long Length { get; }

		public long Offset { get; }

		public long RelativeOffset { get; }

		public IDataType Type
		{
			get; }

		public ILargeChunkedArray<T> Data { get; }

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
				throw new ArgumentNullException(nameof(underlyingBuffer));
			}

			CheckBufferBounds(offset, length, offset + length, underlyingBuffer.Length);

			Length = length;
			RelativeOffset = offset;
			Offset = offset + underlyingBuffer.Offset;

			Data = underlyingBuffer.Data;
			Type = underlyingBuffer.Type;
			_underlyingBuffer = underlyingBuffer;
			_underlyingRootBuffer = underlyingBuffer.GetUnderlyingRootBuffer() == null ? underlyingBuffer : underlyingBuffer.GetUnderlyingRootBuffer();
		}

		/// <summary>
		/// Create a data buffer of a certain large chunked array.
		/// </summary>
		/// <param name="data">The large chunked array data.</param>
		/// <param name="underlyingType">The underlying data type (inferred if not given explicitly).</param>
		public DataBuffer(ILargeChunkedArray<T> data, IDataType underlyingType = null) : this(data, 0L, data?.Length ?? 0L, underlyingType)
		{
		}

		/// <summary>
		/// Create a data buffer of a certain large chunked array.
		/// </summary>
		/// <param name="data">The large chunked array data.</param>
		/// <param name="underlyingType">The underlying data type (inferred if not given explicitly).</param>
		/// <param name="offset">The offset relative to the data array.</param>
		/// <param name="length">The length this buffer should have.</param>
		public DataBuffer(ILargeChunkedArray<T> data, long offset, long length, IDataType underlyingType = null)
		{
			if (data == null)
			{
				throw new ArgumentNullException(nameof(data));
			}

			CheckBufferBounds(offset, length, offset + length, data.Length);

			Data = data;
			Length = length;
			RelativeOffset = offset;
			Offset = offset;

			Type = InferDataType(underlyingType);
		}

		/// <summary>
		/// Create a data buffer of a certain array.
		/// </summary>
		/// <param name="data">The data array.</param>
		/// <param name="underlyingType">The underlying type (inferred if not explicitly given).</param>
		public DataBuffer(T[] data, IDataType underlyingType = null) : this(data, 0L, data?.Length ?? 0L, underlyingType)
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
				throw new ArgumentNullException(nameof(data));
			}

			CheckBufferBounds(offset, length, offset + length, data.Length);

			Data = new LargeChunkedArray<T>(data);
			Length = length;
			RelativeOffset = offset;
			Offset = offset;

			Type = InferDataType(underlyingType);
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

			Length = length;
			Data = new LargeChunkedArray<T>(length);

			Type = InferDataType(underlyingType);
		}

		/// <summary>
		/// Copy constructor.
		/// </summary>
		/// <param name="other">The buffer to copy.</param>
		public DataBuffer(DataBuffer<T> other)
		{
			_underlyingBuffer = other._underlyingBuffer;
			_underlyingRootBuffer = other._underlyingRootBuffer;
			Type = other.Type;
			Data = other.Data;
			Offset = other.Offset;
			RelativeOffset = other.RelativeOffset;
			Length = other.Length;
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

		public object DeepCopy()
		{
			return new DataBuffer<T>((ILargeChunkedArray<T>) Data.DeepCopy(), Type);
		}

		public T GetValue(long index)
		{
			return Data.GetValue(Offset + index);
		}

		public TOther GetValueAs<TOther>(long index)
		{
			return (TOther) Convert.ChangeType(Data.GetValue(Offset + index), typeof(TOther));
		}

		public IDataBuffer<T> GetValues(long startIndex, long length)
		{
			return new DataBuffer<T>(this, startIndex, length);
		}

		public IDataBuffer<TOther> GetValuesAs<TOther>(long startIndex, long length)
		{
			LargeChunkedArray<TOther> otherData = new LargeChunkedArray<TOther>(length);

			otherData.FillWith(Data, Offset + startIndex, 0L, length);

			return new DataBuffer<TOther>(otherData, 0L, length);
		}

		public ILargeChunkedArray<T> GetValuesArray(long startIndex, long length)
		{
			LargeChunkedArray<T> valuesArray = new LargeChunkedArray<T>(length);

			valuesArray.FillWith(Data, Offset + startIndex, 0L, length);

			return valuesArray;
		}

		public ILargeChunkedArray<TOther> GetValuesArrayAs<TOther>(long startIndex, long length)
		{
			LargeChunkedArray<TOther> valuesArray = new LargeChunkedArray<TOther>(length);

			valuesArray.FillWith(Data, Offset + startIndex, 0L, length);

			return valuesArray;
		}

		public void SetValue(T value, long index)
		{
			Data.SetValue(value, index + Offset);
		}

		public void SetValues(IDataBuffer<T> buffer, long sourceStartIndex, long destStartIndex, long length)
		{
			Data.FillWith(buffer.Data, sourceStartIndex + buffer.Offset, destStartIndex + Offset, length);
		}

		public void SetValues(T[] values, long sourceStartIndex, long destStartIndex, long length)
		{
			Data.FillWith(values, sourceStartIndex, destStartIndex + Offset, length);
		}

		public IDataBuffer<T> GetUnderlyingBuffer()
		{
			return _underlyingBuffer;
		}

		public IDataBuffer<T> GetUnderlyingRootBuffer()
		{
			return _underlyingRootBuffer;
		}

		public IEnumerator<T> GetEnumerator()
		{
			for (long i = 0; i < Length; i++)
			{
				yield return Data.GetValue(i);
			}
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			for (long i = 0; i < Length; i++)
			{
				yield return Data.GetValue(i);
			}
		}

		public override string ToString()
		{
			return "$databuffer of type {Type} and size {Length}";
		}
	}
}
