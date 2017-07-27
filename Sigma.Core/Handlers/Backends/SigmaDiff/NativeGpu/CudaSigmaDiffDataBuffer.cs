/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp;
using DiffSharp.Backend;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using Sigma.Core.Data;
using Sigma.Core.Persistence;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	[Serializable]
	public class CudaSigmaDiffDataBuffer<T> : SigmaDiffDataBuffer<T>, ISerialisationNotifier where T : struct
	{
		[NonSerialized]
		internal CudaDeviceVariable<T> CudaBuffer;
		[NonSerialized]
		internal CudaContext CudaContext;

		private int _cudaContextDeviceId;

		[NonSerialized]
		private SizeT _cudaZero;
		[NonSerialized]
		private SizeT _cudaOffsetBytes, _cudaLengthBytes;

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(IDataBuffer<T> underlyingBuffer, long offset, long length, long backendTag, CudaContext cudaContext) : base(underlyingBuffer, offset, length, backendTag)
		{
			InitialiseCudaBuffer(cudaContext, Data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(T[] data, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(data, backendTag, underlyingType)
		{
			InitialiseCudaBuffer(cudaContext, Data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(T[] data, long offset, long length, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(data, offset, length, backendTag, underlyingType)
		{
			InitialiseCudaBuffer(cudaContext, Data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(long length, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(length, backendTag, underlyingType)
		{
			InitialiseCudaBuffer(cudaContext, Data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(DataBuffer<T> other, long backendTag, CudaContext cudaContext) : base(other, backendTag)
		{
			InitialiseCudaBuffer(cudaContext, Data, Offset, Length);
		}

		/// <summary>
		/// Shallow copy constructor with the same system- and device memory buffers.
		/// </summary>
		/// <param name="other"></param>
		internal CudaSigmaDiffDataBuffer(CudaSigmaDiffDataBuffer<T> other) : base(other, other.BackendTag)
		{
			CudaBuffer = other.CudaBuffer;
			CudaContext = other.CudaContext;

			_cudaOffsetBytes = other._cudaOffsetBytes;
			_cudaLengthBytes = other._cudaLengthBytes;
		}

		private void InitialiseCudaBuffer(CudaContext cudaContext, T[] data, long offset, long length)
		{
			if (cudaContext == null) throw new ArgumentNullException(nameof(cudaContext));

			_cudaZero = new SizeT(0);
			_cudaOffsetBytes = new SizeT(offset * Type.SizeBytes);
			_cudaLengthBytes = new SizeT(length * Type.SizeBytes);

			CudaContext = cudaContext;
			CudaBuffer = new CudaDeviceVariable<T>(cudaContext.AllocateMemory(_cudaLengthBytes));
			CopyFromHostToDevice();

			_cudaContextDeviceId = cudaContext.DeviceId;
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			CudaContext restoredContext = CudaFloat32Handler.GetContextForDeviceId(_cudaContextDeviceId);

			InitialiseCudaBuffer(restoredContext, Data, Offset, Length);
		}

		internal void CopyFromHostToDevice()
		{
			CudaBuffer.CopyToDevice(Data, _cudaOffsetBytes, _cudaZero, _cudaLengthBytes);
		}

		internal void CopyFromDeviceToHost()
		{
			CudaBuffer.CopyToHost(Data, _cudaZero, _cudaOffsetBytes, _cudaLengthBytes);
		}

		/// <inheritdoc />
		protected override Util.ISigmaDiffDataBuffer<T> _InternalDeepCopy(T[] copyData)
		{
			return new CudaSigmaDiffDataBuffer<T>(copyData, 0L, Length, BackendTag, CudaContext, Type);
		}

		/// <inheritdoc />
		protected override Util.ISigmaDiffDataBuffer<T> _InternalShallowCopy()
		{
			return new CudaSigmaDiffDataBuffer<T>(this);
		}

		/// <inheritdoc />
		public override Util.ISigmaDiffDataBuffer<T> GetStackedValues(int totalRows, int totalCols, int rowStart, int rowFinish, int colStart, int colFinish)
		{
			int colLength = colFinish - colStart + 1;
			int newSize = (rowFinish - rowStart + 1) * colLength;
			Backend<T> backendHandle = SigmaDiffSharpBackendProvider.Instance.GetBackend<T>(BackendTag).BackendHandle;
			CudaSigmaDiffDataBuffer<T> values = (CudaSigmaDiffDataBuffer<T>) backendHandle.CreateDataBuffer(backendHandle.CreateUninitialisedArray(newSize));

			for (int m = rowStart; m <= rowFinish; m++)
			{
				long sourceIndex = Offset + m * totalCols + colStart;
				long destinationIndex = (m - rowStart) * colLength;

				Buffer.BlockCopy(Data, (int)(sourceIndex * Type.SizeBytes), values.Data, (int)(destinationIndex * Type.SizeBytes), colLength * Type.SizeBytes);
			}

			values.CopyFromHostToDevice();

			return values;
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}
	}
}
