/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Runtime.CompilerServices;
using DiffSharp.Interop.Float32;
using ManagedCuda;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeGpu;
using Sigma.Core.Utils;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	[Serializable]
	public class CudaFloat32Handler : DiffSharpFloat32Handler<CudaFloat32NDArray, CudaFloat32Number>
	{
		/// <summary>
		/// The id of the GPU / CUDA device used in this handler.
		/// </summary>
		public int DeviceId { get; }

		private readonly CudaFloat32BackendHandle _cudaBackendHandle;

		public CudaFloat32Handler(int deviceId = 0) : base(new CudaFloat32BackendHandle(deviceId, backendTag: -1))
		{
			_cudaBackendHandle = (CudaFloat32BackendHandle)DiffsharpBackendHandle;
			DeviceId = _cudaBackendHandle.CudaContext.DeviceId;

			RegisterContext(_cudaBackendHandle.CudaContext, _cudaBackendHandle.CudaStream);
		}

		/// <summary>The underlying data type processed and used in this computation handler.</summary>
		public override IDataType DataType => DataTypes.Float32;

		internal void BindToContext()
		{
			_cudaBackendHandle.BindToContext();
		}

		protected new CudaFloat32NDArray InternaliseArray(object array)
		{
			return AssignTag((CudaFloat32NDArray)array);
		}

		protected new CudaFloat32Number InternaliseNumber(object number)
		{
			return (CudaFloat32Number)number;
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public override void OnDeserialised()
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override void InitAfterDeserialisation(INDArray array)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override long GetSizeBytes(params INDArray[] arrays)
		{
			long totalSizeBytes = 0L;

			foreach (INDArray array in arrays)
			{
				long sizeBytes = 64L; // let's just assume this many bytes of base fluff, I really have no idea (but it has to be more than the CPU version because of all the CUDA handles)

				sizeBytes += array.Length * DataType.SizeBytes;
				sizeBytes += (array.Shape.Length) * 8L * 2;

				totalSizeBytes += sizeBytes;
			}

			return totalSizeBytes;
		}

		/// <inheritdoc />
		public override bool IsOwnFormat(INDArray array)
		{
			return array is CudaFloat32NDArray;
		}

		/// <inheritdoc />
		public override bool IsInterchangeable(IComputationHandler otherHandler)
		{
			return this.GetType() == otherHandler.GetType();
		}

		/// <inheritdoc />
		public override INDArray NDArray(params long[] shape)
		{
			long backendTag = _cudaBackendHandle.BackendTag;

			return AssignTag(new CudaFloat32NDArray(new CudaSigmaDiffDataBuffer<float>(ArrayUtils.Product(shape), backendTag, _cudaBackendHandle.CudaContext), shape));
		}

		/// <inheritdoc />
		public override INDArray NDArray<TOther>(TOther[] values, params long[] shape)
		{
			long backendTag = _cudaBackendHandle.BackendTag;
			float[] convertedValues = new float[values.Length];
			Type floatType = typeof(float);

			for (int i = 0; i < values.Length; i++)
			{
				convertedValues[i] = (float)System.Convert.ChangeType(values[i], floatType);
			}

			return AssignTag(new CudaFloat32NDArray(new CudaSigmaDiffDataBuffer<float>(convertedValues, backendTag, _cudaBackendHandle.CudaContext), shape));
		}

		/// <inheritdoc />
		public override INumber Number(object value)
		{
			return new CudaFloat32Number((float)System.Convert.ChangeType(value, typeof(float)), _cudaBackendHandle.CudaContext).SetAssociatedHandler(this);
		}

		/// <inheritdoc />
		public override IDataBuffer<T> DataBuffer<T>(T[] values)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INDArray AsNDArray(INumber number)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INumber AsNumber(INDArray array, params long[] indices)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override bool CanConvert(INDArray array, IComputationHandler otherHandler)
		{
			return otherHandler.DataType.BaseUnderlyingType == DataType.BaseUnderlyingType && otherHandler.DataType.SizeBytes >= DataType.SizeBytes;
		}

		/// <inheritdoc />
		public override INDArray Convert(INDArray array, IComputationHandler otherHandler)
		{
			return ConvertInternal(array);
		}

		/// <inheritdoc />
		public override void Fill(INDArray filler, INDArray arrayToFill)
		{
			CudaSigmaDiffDataBuffer<float> fillerData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(filler).Data;
			CudaSigmaDiffDataBuffer<float> arrayToFillData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(arrayToFill).Data;

			arrayToFillData.SetValues(fillerData.Data, fillerData.Offset, 0, Math.Min(arrayToFill.Length, filler.Length));
			arrayToFillData.CopyFromHostToDevice();
		}

		/// <inheritdoc />
		public override void Fill(INDArray filler, INDArray arrayToFill, long[] sourceBeginIndices, long[] sourceEndIndices, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			CudaSigmaDiffDataBuffer<float> fillerData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(filler).Data;
			CudaSigmaDiffDataBuffer<float> arrayToFillData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(arrayToFill).Data;

			int sourceOffset = (int)NDArrayUtils.GetFlatIndex(filler.Shape, filler.Strides, sourceBeginIndices);
			int sourceLength = (int)NDArrayUtils.GetFlatIndex(filler.Shape, filler.Strides, sourceEndIndices) - sourceOffset + 1; // +1 because end is inclusive
			int destinationOffset = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationBeginIndices);
			int destinationLength = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationEndIndices) - destinationOffset + 1; // same here

			if (sourceLength < 0) throw new ArgumentOutOfRangeException($"Source begin indices must be smaller than source end indices, but source length was {sourceLength}.");
			if (destinationLength < 0) throw new ArgumentOutOfRangeException($"Destination begin indices must be smaller than destination end indices, but destination length was {destinationLength}.");
			if (sourceLength != destinationLength) throw new ArgumentException($"Source and destination indices length must batch, but source length was {sourceLength} and destination legnth was {destinationLength}.");

			Array.Copy(fillerData.Data, sourceOffset, arrayToFillData.Data, destinationOffset, sourceLength);
			arrayToFillData.CopyFromHostToDevice();
		}

		/// <inheritdoc />
		public override void Fill<T>(T[] filler, INDArray arrayToFill, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			CudaSigmaDiffDataBuffer<float> arrayToFillData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(arrayToFill).Data;

			int destinationOffset = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationBeginIndices);
			int destinationLength = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationEndIndices) - destinationOffset + 1; // +1 because end is inclusive

			if (destinationLength < 0) throw new ArgumentOutOfRangeException($"Destination begin indices must be smaller than destination end indices, but destination length was {destinationLength}.");

			Array.Copy(filler, 0, arrayToFillData.Data, destinationOffset, destinationLength);
			arrayToFillData.CopyFromHostToDevice();
		}

		/// <inheritdoc />
		protected override CudaFloat32NDArray CreateArrayFromHandle(DNDArray handle)
		{
			return new CudaFloat32NDArray(handle);
		}

		/// <inheritdoc />
		protected override CudaFloat32Number CreateNumberFromHandle(DNumber handle)
		{
			return new CudaFloat32Number(handle, _cudaBackendHandle.CudaContext);
		}

		/// <inheritdoc />
		protected override CudaFloat32NDArray ConvertInternal(INDArray array)
		{
			CudaFloat32NDArray @internal = array as CudaFloat32NDArray;

			if (@internal != null) return @internal;

			return new CudaFloat32NDArray(new CudaSigmaDiffDataBuffer<float>(array.GetDataAs<float>(), 0L, array.Length, _cudaBackendHandle.BackendTag, _cudaBackendHandle.CudaContext), (long[])array.Shape.Clone());
		}

		/// <inheritdoc />
		public override void Fill<TOther>(TOther value, INDArray arrayToFill)
		{
			CudaSigmaDiffDataBuffer<float> arrayToFillData = (CudaSigmaDiffDataBuffer<float>)InternaliseArray(arrayToFill).Data;
			float floatValue = (float)System.Convert.ChangeType(value, typeof(float));

			for (int i = 0; i < arrayToFillData.Length; i++)
			{
				arrayToFillData.Data.SetValue(floatValue, i);
			}

			arrayToFillData.CopyFromHostToDevice();
		}

		/// <inheritdoc />
		public override void MarkLimbo(INDArray array)
		{
			CudaFloat32NDArray internalArray = InternaliseArray(array);

			_cudaBackendHandle.MarkLimbo(internalArray.UnsafeRawData);
		}

		/// <inheritdoc />
		public override void FreeLimbo(INDArray array)
		{
			CudaFloat32NDArray internalArray = InternaliseArray(array);

			_cudaBackendHandle.FreeLimbo(internalArray.UnsafeRawData);
		}

		/// <inheritdoc />
		protected override bool _RowWiseOptimised(ref CudaFloat32NDArray array, Func<INDArray, INDArray> function)
		{
			if (function == SoftMax)
			{
				CudaFloat32NDArray internalArray = InternaliseArray(array);

				array = CreateArrayFromHandle(DNDArray.CustomOp(internalArray.Handle, new CudaFloat32BackendHandle.CustomOpHandle(CudaFloat32BackendHandle.CustomOpType.RowWiseSoftmax)));

				return true;
			}

			return false;
		}

		private static readonly WeakList<CudaContext> RegisteredContexts;
		private static readonly ConditionalWeakTable<CudaContext, CudaStream> RegisteredStreamsByContext;

		static CudaFloat32Handler()
		{
			RegisteredContexts = new WeakList<CudaContext>();
			RegisteredStreamsByContext = new ConditionalWeakTable<CudaContext, CudaStream>();
		}

		/// <summary>
		/// Register a cuda context for reference via <see cref="GetContextForDeviceId"/> (used for context restoration).
		/// </summary>
		/// <param name="context"></param>
		internal static void RegisterContext(CudaContext context, CudaStream stream)
		{
			if (!RegisteredContexts.Contains(context))
			{
				RegisteredContexts.Add(context);
				RegisteredStreamsByContext.Add(context, stream);
			}
		}

		/// <summary>
		/// Get a CUDA stream for a certain CUDA context.
		/// </summary>
		/// <param name="context">The context.</param>
		/// <returns>The stream.</returns>
		internal static CudaStream GetStreamForContext(CudaContext context)
		{
			CudaStream stream;

			if (RegisteredStreamsByContext.TryGetValue(context, out stream))
			{
				return stream;
			}

			throw new InvalidOperationException($"No stream is registered for context with id {context.DeviceId}.");
		}

		/// <summary>
		/// Get a CUDA context with preference for a certain deviceId, optionally accept any device.
		/// </summary>
		/// <param name="deviceId">The device id.</param>
		/// <param name="acceptOtherDevices">Optionally indicate if any device is acceptable (if preferred deviceId is not available).</param>
		/// <returns></returns>
		internal static CudaContext GetContextForDeviceId(int deviceId, bool acceptOtherDevices = true)
		{
			foreach (CudaContext context in RegisteredContexts)
			{
				if (context.DeviceId == deviceId)
				{
					return context;
				}
			}

			if (acceptOtherDevices && RegisteredContexts.Count > 0)
			{
				return RegisteredContexts.FirstAvailable();
			}

			if (acceptOtherDevices) throw new InvalidOperationException($"No valid cuda context is currently registered.");

			throw new InvalidOperationException($"No valid cuda context with the device id {deviceId} is currently registered " +
												$"(use {nameof(acceptOtherDevices)} flag to accept any devices if the preferred one is unavailable).");
		}
	}
}
