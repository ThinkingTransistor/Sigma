/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Data;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	[Serializable]
	internal class SigmaDiffDataBuffer<T> : DataBuffer<T>, ISigmaDiffDataBuffer<T>
	{
		#region DiffSharp SigmaDiffDataBuffer interop properties

		int ISigmaDiffDataBuffer<T>.Length => (int) Length;

		int ISigmaDiffDataBuffer<T>.Offset => (int) Offset;

		T[] ISigmaDiffDataBuffer<T>.Data => Data;

		T[] ISigmaDiffDataBuffer<T>.SubData => DataBufferSubDataUtils.SubData(Data, (int) Offset, (int) Length);

		#endregion

		public SigmaDiffDataBuffer(IDataBuffer<T> underlyingBuffer, long offset, long length) : base(underlyingBuffer, offset, length)
		{
		}

		public SigmaDiffDataBuffer(T[] data, IDataType underlyingType = null) : base(data, underlyingType)
		{
		}

		public SigmaDiffDataBuffer(T[] data, long offset, long length, IDataType underlyingType = null) : base(data, offset, length, underlyingType)
		{
		}

		public SigmaDiffDataBuffer(long length, IDataType underlyingType = null) : base(length, underlyingType)
		{
		}

		public SigmaDiffDataBuffer(DataBuffer<T> other) : base(other)
		{
		}

		#region DiffSharp SigmaDiffDataBuffer interop methods

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.GetValues(int startIndex, int length)
		{
			return (ISigmaDiffDataBuffer<T>) GetValues(startIndex, length);
		}

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.DeepCopy()
		{
			return (ISigmaDiffDataBuffer<T>) DeepCopy();
		}

		ISigmaDiffDataBuffer<T> ISigmaDiffDataBuffer<T>.ShallowCopy()
		{
			return (ISigmaDiffDataBuffer<T>) ShallowCopy();
		}

		#endregion
	}
}
