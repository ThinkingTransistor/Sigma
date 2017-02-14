/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Data;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// A DataBuffer wrapper for use with the Sigma.DiffSharp library. 
	/// </summary>
	/// <typeparam name="T"></typeparam>
	[Serializable]
	internal class SigmaDiffDataBuffer<T> : DataBuffer<T>, ISigmaDiffDataBuffer<T>
	{
		public long BackendTag { get; set; }

		#region DiffSharp SigmaDiffDataBuffer interop properties

		int ISigmaDiffDataBuffer<T>.Length => (int) Length;

		int ISigmaDiffDataBuffer<T>.Offset => (int) Offset;

		T[] ISigmaDiffDataBuffer<T>.Data => Data;

		T[] ISigmaDiffDataBuffer<T>.SubData => DataBufferSubDataUtils.SubData(Data, (int) Offset, (int) Length);

		#endregion

		public SigmaDiffDataBuffer(IDataBuffer<T> underlyingBuffer, long offset, long length, long backendTag) : base(underlyingBuffer, offset, length)
		{
			BackendTag = backendTag;
		}

		public SigmaDiffDataBuffer(T[] data, long backendTag, IDataType underlyingType = null) : base(data, underlyingType)
		{
			BackendTag = backendTag;
		}

		public SigmaDiffDataBuffer(T[] data, long offset, long length, long backendTag, IDataType underlyingType = null) : base(data, offset, length, underlyingType)
		{
			BackendTag = backendTag;
		}

		public SigmaDiffDataBuffer(long length, long backendTag, IDataType underlyingType = null) : base(length, underlyingType)
		{
			BackendTag = backendTag;
		}

		public SigmaDiffDataBuffer(DataBuffer<T> other, long backendTag) : base(other)
		{
			BackendTag = backendTag;
		}

		public override IDataBuffer<T> GetValues(long startIndex, long length)
		{
			return new SigmaDiffDataBuffer<T>(this, startIndex, length, BackendTag);
		}

		public override IDataBuffer<TOther> GetValuesAs<TOther>(long startIndex, long length)
		{
			return new SigmaDiffDataBuffer<TOther>(GetValuesArrayAs<TOther>(startIndex, length), 0L, length, BackendTag);
		}

		public override object DeepCopy()
		{
			return new SigmaDiffDataBuffer<T>((T[]) Data.Clone(), Offset, Length, BackendTag, Type);
		}

		#region DiffSharp SigmaDiffDataBuffer interop methods

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.GetValues(int startIndex, int length)
		{
			return (ISigmaDiffDataBuffer<T>) GetValues(startIndex, length);
		}

		public ISigmaDiffDataBuffer<T> GetStackedValues(int totalRows, int totalCols, int rowStart, int rowFinish, int colStart, int colFinish)
		{
			int newSize = (rowFinish - rowStart + 1) * (colFinish - colStart + 1);
			SigmaDiffDataBuffer<T> values = new SigmaDiffDataBuffer<T>(new T[newSize], BackendTag);
			int colLength = colFinish - colStart + 1;

			for (int m = rowStart; m <= rowFinish; m++)
			{
				long sourceIndex = Offset + m * totalCols + colStart;
				long destinationIndex = m * (totalCols - 1 - colStart - colFinish) + colStart;

				System.Array.Copy(Data, sourceIndex, values.Data, destinationIndex, colLength);
			}

			return values;
		}

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.DeepCopy()
		{
			T[] copyData = new T[Length];
			System.Array.Copy(Data, Offset, copyData, 0, Length);

			// deep copy only core data for diffsharp
			return new SigmaDiffDataBuffer<T>(copyData, 0L, Length, BackendTag, Type);
		}

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.ShallowCopy()
		{
			return new SigmaDiffDataBuffer<T>(this, BackendTag);
		}

		#endregion
	}
}
