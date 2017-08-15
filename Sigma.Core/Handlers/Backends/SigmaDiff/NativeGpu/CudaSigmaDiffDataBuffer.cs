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
using ManagedCuda.CudaBlas;
using Sigma.Core.Data;
using Sigma.Core.Persistence;
using Sigma.Core.Utils;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	[Serializable]
	public class CudaSigmaDiffDataBuffer<T> : SigmaDiffDataBuffer<T>, ISerialisationNotifier where T : struct
	{
		internal SizeT CudaLengthBytes { get { return _cudaLengthBytes; } }

		[NonSerialized]
		internal CudaContext CudaContext;

		// TODO implement more intelligent host <-> device synchronisation by flagging these whenever meaningful host read / device write access occurs
		private bool _flagHostModified;
		private bool _flagDeviceModified;

		[NonSerialized]
		private bool _initialisedInContext;

		private int _cudaContextDeviceId;

		[NonSerialized]
		private CudaSigmaDiffDataBuffer<T> _underlyingCudaBuffer;
		[NonSerialized]
		private CudaDeviceVariable<T> _cudaBuffer;
		[NonSerialized]
		private SizeT _cudaZero;
		[NonSerialized]
		private SizeT _cudaOffsetBytes, _cudaLengthBytes;

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(IDataBuffer<T> underlyingBuffer, long offset, long length, long backendTag, CudaContext cudaContext) : base(underlyingBuffer, offset, length, backendTag)
		{
			PrepareCudaBuffer(cudaContext, _data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(T[] data, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(data, backendTag, underlyingType)
		{
			PrepareCudaBuffer(cudaContext, _data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(T[] data, long offset, long length, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(data, offset, length, backendTag, underlyingType)
		{
			PrepareCudaBuffer(cudaContext, _data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(long length, long backendTag, CudaContext cudaContext, IDataType underlyingType = null) : base(length, backendTag, underlyingType)
		{
			PrepareCudaBuffer(cudaContext, _data, Offset, Length);
		}

		/// <inheritdoc />
		public CudaSigmaDiffDataBuffer(DataBuffer<T> other, long backendTag, CudaContext cudaContext) : base(other, backendTag)
		{
			PrepareCudaBuffer(cudaContext, _data, Offset, Length);
		}

		/// <summary>
		/// Shallow copy constructor with the same system- and device memory buffers.
		/// </summary>
		/// <param name="other"></param>
		internal CudaSigmaDiffDataBuffer(CudaSigmaDiffDataBuffer<T> other) : base(other, other.BackendTag)
		{
			_cudaBuffer = other._cudaBuffer;
			CudaContext = other.CudaContext;

			_cudaOffsetBytes = other._cudaOffsetBytes;
			_cudaLengthBytes = other._cudaLengthBytes;

			_flagDeviceModified = other._flagDeviceModified;
			_flagHostModified = other._flagHostModified;
		}

		private void PrepareCudaBuffer(CudaContext cudaContext, T[] data, long offset, long length)
		{
			CudaContext = cudaContext;

			_cudaZero = new SizeT(0);
			_cudaOffsetBytes = new SizeT(offset * Type.SizeBytes);
			_cudaLengthBytes = new SizeT(length * Type.SizeBytes);

			_cudaContextDeviceId = cudaContext.DeviceId;
			_underlyingCudaBuffer = GetUnderlyingBuffer() as CudaSigmaDiffDataBuffer<T>;
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			CudaContext restoredContext = CudaFloat32Handler.GetContextForDeviceId(_cudaContextDeviceId);

			PrepareCudaBuffer(restoredContext, Data, Offset, Length);
		}

		internal void InitialiseCudaBuffer(bool copyHostToDevice = true)
		{
			if (CudaContext == null) throw new InvalidOperationException($"Cannot initialise cuda buffer, cuda context is invalid (null).");

			if (_underlyingCudaBuffer != null)
			{
				if (!_underlyingCudaBuffer._initialisedInContext)
				{
					_underlyingCudaBuffer.InitialiseCudaBuffer(copyHostToDevice);
				}

				_cudaBuffer = new CudaDeviceVariable<T>(_underlyingCudaBuffer._cudaBuffer.DevicePointer + _cudaOffsetBytes, _cudaLengthBytes);

				_initialisedInContext = true;
			}
			else
			{
				CudaFloat32BackendHandle backendHandle = (CudaFloat32BackendHandle)SigmaDiffSharpBackendProvider.Instance.GetBackend<T>(BackendTag).BackendHandle;

				bool initialisedToValue;
				_cudaBuffer = backendHandle.AllocateDeviceBuffer(_data, Offset, _cudaLengthBytes, out initialisedToValue);
				_initialisedInContext = true;

				if (copyHostToDevice && !initialisedToValue)
				{
					CopyFromHostToDevice();

					_flagHostModified = false;
				}
			}
		}

		internal CudaDeviceVariable<T> GetContextBuffer()
		{
			if (!_initialisedInContext)
			{
				InitialiseCudaBuffer();
			}

			SynchroniseFromHostToDevice();

			return _cudaBuffer;
		}

		internal CUdeviceptr GetContextPointer()
		{
			return GetContextBuffer().DevicePointer;
		}

		/// <summary>
		/// Called before any operation that reads from the local data.
		/// </summary>
		protected override void OnReadAccess()
		{
			SynchroniseFromDeviceToHost();
		}

		/// <summary>
		/// Called before any operation that writes to the local data.
		/// </summary>
		protected override void OnWriteAccess()
		{
			SynchroniseFromDeviceToHost();

			_flagHostModified = true;
		}

		/// <summary>
		/// Called before any operation that reads from and writes to the local data.
		/// </summary>
		protected override void OnReadWriteAccess()
		{
			SynchroniseFromDeviceToHost();

			_flagHostModified = true;
		}

		internal void FlagDeviceModified()
		{
			_flagDeviceModified = true;
		}

		internal void FlagHostModified()
		{
			_flagHostModified = true;
		}

		internal bool IsHostModified()
		{
			return _flagHostModified;
		}

		internal bool IsDeviceModified()
		{
			return _flagDeviceModified;
		}

		internal bool IsInitialisedInContext()
		{
			return _initialisedInContext;
		}

		internal void SynchroniseFromHostToDevice()
		{
			if (_flagHostModified)
			{
				if (_flagDeviceModified)
				{
					throw new InvalidOperationException($"Unable to synchronise buffers from host to device, both device and host buffers are marked modified.");
				}

				CopyFromHostToDevice();

				_flagHostModified = false;
			}
		}

		internal void SynchroniseFromDeviceToHost()
		{
			if (_flagDeviceModified)
			{
				if (_flagHostModified)
				{
					throw new InvalidOperationException($"Unable to synchronise buffers from device to host, both device and host buffers are marked modified.");
				}

				CopyFromDeviceToHost();

				_flagDeviceModified = false;
			}
		}

		internal void CopyFromHostToDevice()
		{
			if (!_initialisedInContext)
			{
				InitialiseCudaBuffer(copyHostToDevice: false);
			}

			CudaFloat32BackendHandle backendHandle = (CudaFloat32BackendHandle)SigmaDiffSharpBackendProvider.Instance.GetBackend<T>(BackendTag).BackendHandle;
			backendHandle.MarkDeviceBufferModified((float[])(object)_data);

			_cudaBuffer.CopyToDevice(_data, _cudaOffsetBytes, _cudaZero, _cudaLengthBytes);
		}

		internal void CopyFromDeviceToHost()
		{
			if (!_initialisedInContext)
			{
				InitialiseCudaBuffer(copyHostToDevice: false);
			}

			_cudaBuffer.CopyToHost(_data, _cudaZero, _cudaOffsetBytes, _cudaLengthBytes);
		}

		/// <inheritdoc />
		protected override Util.ISigmaDiffDataBuffer<T> _InternalDeepCopy()
		{
			T[] copyData = SigmaDiffSharpBackendProvider.Instance.GetBackend<T>(BackendTag).BackendHandle.CreateUninitialisedArray((int)Length);
			CudaSigmaDiffDataBuffer<T> copy = new CudaSigmaDiffDataBuffer<T>(copyData, 0L, Length, BackendTag, CudaContext, Type);

			if (_initialisedInContext && !_flagHostModified)
			{
				copy.InitialiseCudaBuffer(copyHostToDevice: false);

				copy._cudaBuffer.AsyncCopyToDevice(_cudaBuffer, CudaFloat32Handler.GetStreamForContext(CudaContext));

				copy._flagHostModified = false;
				copy._flagDeviceModified = true;
			}
			else
			{
				OnReadAccess();

				Buffer.BlockCopy(_data, (int)(Offset * Type.SizeBytes), copyData, 0, (int)(Length * Type.SizeBytes));

				copy._flagHostModified = true;
			}

			return copy;
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
			CudaSigmaDiffDataBuffer<T> values = (CudaSigmaDiffDataBuffer<T>)backendHandle.CreateDataBuffer(backendHandle.CreateUninitialisedArray(newSize));

			if (_initialisedInContext)
			{
				SynchroniseFromDeviceToHost();
			}

			for (int m = rowStart; m <= rowFinish; m++)
			{
				long sourceIndex = Offset + m * totalCols + colStart;
				long destinationIndex = (m - rowStart) * colLength;

				Buffer.BlockCopy(_data, (int)(sourceIndex * Type.SizeBytes), values.Data, (int)(destinationIndex * Type.SizeBytes), colLength * Type.SizeBytes);
			}

			if (_initialisedInContext)
			{
				values.SynchroniseFromHostToDevice();
			}

			return values;
		}

		/// <inheritdoc />
		public override IDataBuffer<T> GetValues(long startIndex, long length)
		{
			if (_initialisedInContext && !_flagHostModified)
			{
				CudaSigmaDiffDataBuffer<T> vData = new CudaSigmaDiffDataBuffer<T>(_data, startIndex, length, BackendTag, CudaContext);

				vData.InitialiseCudaBuffer(copyHostToDevice: false);

				vData._cudaBuffer.AsyncCopyToDevice(_cudaBuffer.DevicePointer, new SizeT(startIndex * sizeof(float)), vData._cudaZero, vData._cudaLengthBytes,
					CudaFloat32Handler.GetStreamForContext(CudaContext));

				vData._flagHostModified = false;
				vData._flagDeviceModified = false;

				return vData;
			}
			else
			{
				OnReadAccess();

				return new CudaSigmaDiffDataBuffer<T>(this, startIndex, length, BackendTag, CudaContext);
			}
		}

		/// <inheritdoc />
		public override IDataBuffer<TOther> GetValuesAs<TOther>(long startIndex, long length)
		{
			OnReadAccess();

			return new CudaSigmaDiffDataBuffer<TOther>(GetValuesArrayAs<TOther>(startIndex, length), 0L, length, BackendTag, CudaContext);
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}
	}
}
