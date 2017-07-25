/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using Sigma.Core.Data;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaSigmaDiffDataBuffer<T> : SigmaDiffDataBuffer<T> where T : struct
	{
		internal CudaDeviceVariable<T> CudaBuffer;
		internal CudaContext CudaContext;

		private readonly SizeT _cudaZero = new SizeT(0);
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

		private void InitialiseCudaBuffer(CudaContext cudaContext, T[] data, long offset, long length)
		{
			if (cudaContext == null) throw new ArgumentNullException(nameof(cudaContext));

			_cudaOffsetBytes = new SizeT(offset * Type.SizeBytes);
			_cudaLengthBytes = new SizeT(length * Type.SizeBytes);

			CudaBuffer = new CudaDeviceVariable<T>(cudaContext.AllocateMemory(new SizeT(length)));
			CopyFromHostToDevice();

			CudaContext = cudaContext;
		}

		internal void CopyFromHostToDevice()
		{
			CudaBuffer.CopyToDevice(Data, _cudaOffsetBytes, _cudaZero, _cudaLengthBytes / Type.SizeBytes); // TODO is it bytes or just length? API doc says bytes, error says nope
		}

		internal void CopyFromDeviceToHost()
		{
			CudaBuffer.CopyToHost(Data, _cudaZero, _cudaOffsetBytes, _cudaLengthBytes);
		}
	}
}
