/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using DiffSharp.Backend;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.VectorTypes;
using Microsoft.FSharp.Core;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaFloat32BackendHandle : DiffSharpBackendHandle<float>
	{
		internal CudaBlas CudaBlasHandle;
		internal readonly CudaContext CudaContext;
		internal readonly CudaStream CudaStream;

		private readonly IDictionary<object, CudaDeviceVariable<float>> _allocatedDeviceBuffers;
		private readonly ConditionalWeakTable<float[], object> _preInitialisedHostDatas;

		private const int BlocksPerGridDimension = 65535;
		private const int ThreadsPerBlock = 256; // TODO if this constant is changed, the sigmakernels.cu file has to be updated and recompiled with a different number for curandStates
		private readonly object _throwawayObject = new object();
		private readonly CUmodule _kernelModule;
		private readonly IDictionary<int, int> _bufferReferenceCounts;
		private readonly ISet<float[]> _pendingBufferDisposals;
		private readonly IDictionary<string, CudaKernel> _loadedKernels;

		public CudaFloat32BackendHandle(int deviceId, long backendTag) : base(backendTag)
		{
			CudaContext = new CudaContext(deviceId);
			CudaStream = new CudaStream();

			_kernelModule = CudaContext.LoadModulePTX("sigmakernels.ptx");
			_loadedKernels = LoadKernels(_kernelModule);

			_allocatedDeviceBuffers = new Dictionary<object, CudaDeviceVariable<float>>();
			_preInitialisedHostDatas = new ConditionalWeakTable<float[], object>();
			_bufferReferenceCounts = new ConcurrentDictionary<int, int>();
			_pendingBufferDisposals = new HashSet<float[]>();

			BindToContext();
		}

		private IDictionary<string, CudaKernel> LoadKernels(CUmodule kernelModule)
		{
			IDictionary<string, CudaKernel> loadedKernels = new Dictionary<string, CudaKernel>();

			loadedKernels.Add("InitialiseRandomStates", new CudaKernel("_Z22InitialiseRandomStatesi", kernelModule, CudaContext));
			loadedKernels.Add("FillWithProbabilityMask_V", new CudaKernel("_Z25FillWithProbabilityMask_VPffi", kernelModule, CudaContext));

			loadedKernels.Add("Sub_V_S", new CudaKernel("_Z7Sub_V_SPKffPfi", kernelModule, CudaContext));
			loadedKernels.Add("Sub_S_V", new CudaKernel("_Z7Sub_S_VfPfS_i", kernelModule, CudaContext));
			loadedKernels.Add("Add_V_S", new CudaKernel("_Z7Add_V_SPKffPfi", kernelModule, CudaContext));
			loadedKernels.Add("Add_V_V", new CudaKernel("_Z15Add_V_V_InPlacePKfiPfii", kernelModule, CudaContext));
			loadedKernels.Add("Mul_Had_V_V", new CudaKernel("_Z11Mul_Had_V_VPKfS0_Pfi", kernelModule, CudaContext));
			loadedKernels.Add("Div_S_V", new CudaKernel("_Z7Div_S_VfPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Div_V_V", new CudaKernel("_Z7Div_V_VPKfS0_Pfi", kernelModule, CudaContext));

			loadedKernels.Add("Exp_V", new CudaKernel("_Z5Exp_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Log_V", new CudaKernel("_Z5Log_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Sqrt_V", new CudaKernel("_Z6Sqrt_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Sign_V", new CudaKernel("_Z6Sign_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Rel_V", new CudaKernel("_Z5Rel_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Sigmoid_V", new CudaKernel("_Z9Sigmoid_VPKfPfi", kernelModule, CudaContext));

			loadedKernels.Add("Sum_V", new CudaKernel("_Z5Sum_VPKfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Sum_M_Rowwise", new CudaKernel("_Z13Sum_M_RowwisePKfiiiPfi", kernelModule, CudaContext));
			loadedKernels.Add("Add_M_Rowwise_V_InPlace", new CudaKernel("_Z23Add_M_Rowwise_V_InPlacePKfiiiPfi", kernelModule, CudaContext));

			loadedKernels.Add("Softmax_Rowwise_M", new CudaKernel("_Z17Softmax_Rowwise_MPKfPfS1_S1_iiiS1_i", kernelModule, CudaContext));
			loadedKernels.Add("Softmax_Rowwise_M_Backward", new CudaKernel("_Z26Softmax_Rowwise_M_BackwardPKfS0_S0_S0_S0_S0_Pfiiii", kernelModule, CudaContext));

			loadedKernels.Add("Permute_M", new CudaKernel("_Z9Permute_MPKfS0_S0_PfS0_ii", kernelModule, CudaContext));
			loadedKernels.Add("RepeatReshapeCopy_V_MRows", new CudaKernel("_Z25RepeatReshapeCopy_V_MRowsPKfPfiii", kernelModule, CudaContext));

			return loadedKernels;
		}

		private void RunKernel(string kernelName, int threadCount, params object[] kernelParameters)
		{
			RunKernel(kernelName, threadCount, 0, kernelParameters);
		}

		private void RunKernel(string kernelName, int threadCount, uint sharedMemoryBytes, params object[] kernelParameters)
		{
			if (!_loadedKernels.ContainsKey(kernelName))
			{
				throw new InvalidOperationException($"Unable to run kernel, kernel with name {kernelName} is not loaded.");
			}

			CudaKernel kernel = _loadedKernels[kernelName];

			int primaryGridDimensions = Math.Min(BlocksPerGridDimension, (threadCount + ThreadsPerBlock - 1) / ThreadsPerBlock);
			int secondaryGridDimensions = (threadCount + primaryGridDimensions * ThreadsPerBlock - 1) / (primaryGridDimensions * ThreadsPerBlock);

			if (secondaryGridDimensions > BlocksPerGridDimension)
			{
				throw new InvalidOperationException($"Attempted to spawn unsupported amount of threads: {threadCount}, " +
													$"maximum per block is {ThreadsPerBlock} and blocks per grid dimensions (x, y) is {BlocksPerGridDimension}.");
			}

			kernel.BlockDimensions = ThreadsPerBlock;
			kernel.GridDimensions = new dim3(primaryGridDimensions, secondaryGridDimensions, 1);
			kernel.DynamicSharedMemory = sharedMemoryBytes;

			kernel.RunAsync(CudaStream.Stream, kernelParameters);
		}

		internal void BindToContext()
		{
			CudaContext.SetCurrent();
			CudaBlasHandle = new CudaBlas();
			CudaBlasHandle.Stream = CudaStream.Stream;

			RunKernel("InitialiseRandomStates", ThreadsPerBlock, Stopwatch.GetTimestamp());
		}

		/// <summary>
		/// Allocate a CUDA buffer on the used device for a certain host array.
		/// </summary>
		/// <typeparam name="T">The buffer type (only float32 supported here).</typeparam>
		/// <param name="hostData">The host version this data.</param>
		/// <param name="hostOffset">The offset within the host array.</param>
		/// <param name="requestedLengthBytes">The length in bytes as a SizeT struct (if allocation is required).</param>
		/// <param name="initialisedToValue">Indicate whether the allocated device buffer was already pre-initialised to the value requested by a preceding Create call.</param>
		/// <returns>A CUDA buffer corresponding to the host array of the required size (cached if already exists, otherwise newly allocated).</returns>
		internal CudaDeviceVariable<T> AllocateDeviceBuffer<T>(T[] hostData, long hostOffset, SizeT requestedLengthBytes, out bool initialisedToValue)
			where T : struct
		{
			// TODO this casting and type checking is absolutely horribly, need to improve the way the data buffer accesses this so that it can be either truly dynamic or fixed type
			if (typeof(T) != typeof(float)) throw new InvalidOperationException($"{nameof(CudaFloat32BackendHandle)} can only allocate float32 device buffers, given type {typeof(T)} is not valid.");

			IncreaseReferenceCount(hostData);

			// The caching here works because we're essentially tagging along with the in-system memory / host caching done by defaul tin DiffSharpBackendHandle<T>.
			// The idea is that every host array has a corresponding device buffer, and because the host arrays are already reused as necessary,
			//  the device buffers are too as they are weakly associated with the host arrays. 
			//  This (in theory) also automatically takes care of "freeing" device buffers when the owning host array is finalised.
			CudaDeviceVariable<float> deviceBuffer;
			if (_allocatedDeviceBuffers.TryGetValue(hostData, out deviceBuffer))
			{
				object throwaway;
				initialisedToValue = _preInitialisedHostDatas.TryGetValue((float[])(object)hostData, out throwaway);

				if (deviceBuffer.SizeInBytes == requestedLengthBytes)
				{
					return (CudaDeviceVariable<T>)(object)deviceBuffer;
				}
				else
				{
					return new CudaDeviceVariable<T>(deviceBuffer.DevicePointer + hostOffset * sizeof(float), requestedLengthBytes);
				}
			}

			initialisedToValue = false;

			SizeT totalSizeBytes = new SizeT(hostData.Length * sizeof(float));
			deviceBuffer = new CudaDeviceVariable<float>(CudaContext.AllocateMemory(totalSizeBytes), true, totalSizeBytes);
			_allocatedDeviceBuffers.Add(hostData, deviceBuffer);

			if (deviceBuffer.SizeInBytes != requestedLengthBytes)
			{
				return new CudaDeviceVariable<T>(deviceBuffer.DevicePointer + hostOffset * sizeof(float), requestedLengthBytes);
			}

			return (CudaDeviceVariable<T>)(object)deviceBuffer;
		}

		/// <summary>
		/// Increase the reference count for a certain host data (notify that a new device buffer is referencing this host buffer).
		/// Note: This is used for automatically disposing stale buffers
		/// </summary>
		/// <typeparam name="T">The host data type.</typeparam>
		/// <param name="hostData">The host data to which a new device "reference" has been established.</param>
		internal void IncreaseReferenceCount<T>(T[] hostData) where T : struct
		{
			int bufferReferenceId = hostData.GetHashCode(); // this seems particularly unsafe

			int bufferReferenceCount;
			if (!_bufferReferenceCounts.TryGetValue(bufferReferenceId, out bufferReferenceCount))
			{
				_bufferReferenceCounts[bufferReferenceId] = 1;
			}
			else
			{
				_bufferReferenceCounts[bufferReferenceId] = bufferReferenceCount + 1;
			}
		}

		/// <summary>
		/// Notify that a device buffer associated with a certain host array is to be freed.
		/// Note: The actual underlying device buffer is not immediately freed. Rather, a reference counter ensures that no other buffers reference this buffer's memory.
		/// </summary>
		/// <param name="hostData">The host data.</param>
		internal void NotifyFreeDeviceBuffer(float[] hostData)
		{
			int bufferReferenceId = hostData.GetHashCode();

			if (_bufferReferenceCounts.ContainsKey(bufferReferenceId))
			{
				int bufferReferenceCount = _bufferReferenceCounts[bufferReferenceId] - 1;

				if (bufferReferenceCount <= 0)
				{
					CudaDeviceVariable<float> buffer;

					if (_allocatedDeviceBuffers.TryGetValue(hostData, out buffer))
					{
						lock (_pendingBufferDisposals)
						{
							_pendingBufferDisposals.Add(hostData);
						}
					}

					_bufferReferenceCounts.Remove(bufferReferenceId);
				}
				else
				{
					_bufferReferenceCounts[bufferReferenceId] = bufferReferenceCount;
				}
			}
		}

		internal override void TransferSessionBuffers()
		{
			base.TransferSessionBuffers();

			lock (_pendingBufferDisposals)
			{
				foreach (float[] pendingHostData in _pendingBufferDisposals)
				{
					int bufferReferenceId = pendingHostData.GetHashCode();

					if (!IsRegistered(pendingHostData) && !_bufferReferenceCounts.ContainsKey(bufferReferenceId))
					{
						CudaDeviceVariable<float> pendingBuffer;

						if (_allocatedDeviceBuffers.TryGetValue(pendingHostData, out pendingBuffer))
						{
							pendingBuffer.Dispose();

							_allocatedDeviceBuffers.Remove(pendingHostData);
						}
					}
				}

				_pendingBufferDisposals.Clear();
			}
		}

		/// <summary>
		/// Mark the device buffer (corresponding to a certain host array) modified.
		/// Note: This is used for optimising host / device synchronisation on initialisation.
		/// </summary>
		/// <param name="hostData">The corresponding host data.</param>
		internal void MarkDeviceBufferModified(float[] hostData)
		{
			_preInitialisedHostDatas.Remove(hostData);
		}

		/// <summary>
		/// Called when an uninitialised value array is "created" (from cache or allocated).
		/// </summary>
		/// <param name="array">The array.</param>
		protected override void OnUninitialisedArrayCreated(float[] array)
		{
			CudaDeviceVariable<float> deviceBuffer;
			if (_allocatedDeviceBuffers.TryGetValue(array, out deviceBuffer))
			{
				object throwaway;
				if (!_preInitialisedHostDatas.TryGetValue(array, out throwaway))
				{
					_preInitialisedHostDatas.Add(array, _throwawayObject);
				}
			}
		}

		/// <summary>
		/// Called when a value array is "created" (from cache or allocated).
		/// </summary>
		/// <param name="array">The array.</param>
		/// <param name="initialValue">The initial value.</param>
		protected override unsafe void OnValueArrayCreated(float[] array, float initialValue)
		{
			CudaDeviceVariable<float> deviceBuffer;
			if (_allocatedDeviceBuffers.TryGetValue(array, out deviceBuffer))
			{
				deviceBuffer.MemsetAsync(*(uint*)&initialValue, CudaStream.Stream);

				object throwaway;
				if (!_preInitialisedHostDatas.TryGetValue(array, out throwaway))
				{
					_preInitialisedHostDatas.Add(array, _throwawayObject);
				}
			}
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> CreateDataBuffer(float[] values)
		{
			return new CudaSigmaDiffDataBuffer<float>(values, BackendTag, CudaContext);
		}

		private CudaSigmaDiffDataBuffer<float> _InternalInternalise(ISigmaDiffDataBuffer<float> value)
		{
			return (CudaSigmaDiffDataBuffer<float>)value;
		}

		private CudaSigmaDiffDataBuffer<float> _InternalInternalise(ShapedDataBufferView<float> value)
		{
			return (CudaSigmaDiffDataBuffer<float>)value.DataBuffer;
		}

		internal class CustomOpHandle
		{
			internal CustomOpType Type { get; }
			private IDictionary<string, object> AdditionalInfo { get; }

			internal CustomOpHandle(CustomOpType type)
			{
				Type = type;
				AdditionalInfo = new Dictionary<string, object>();
			}

			internal void AttachInfo(string identifier, object obj)
			{
				AdditionalInfo.Add(identifier, obj);
			}

			internal T GetInfo<T>(string identifier)
			{
				return (T)AdditionalInfo[identifier];
			}
		}

		internal enum CustomOpType
		{
			RowWiseSoftmax
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> CustomOp_DM_Forward(ShapedDataBufferView<float> a, object customInfo)
		{
			if (!(customInfo is CustomOpHandle))
			{
				throw new InvalidOperationException($"Cannot invoke {nameof(CustomOp_DM_Forward)} with invalid custom info of type {customInfo.GetType()} (must be of type {nameof(CustomOpHandle)}).");
			}

			CustomOpHandle op = (CustomOpHandle)customInfo;

			if (!Enum.IsDefined(typeof(CustomOpType), op.Type))
			{
				throw new NotImplementedException($"Custom op {op} is not supported in {nameof(CustomOp_DM_Forward)}.");
			}

			int len = a.Length;
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			if (op.Type == CustomOpType.RowWiseSoftmax)
			{
				int colsNextPowerOf2 = ArrayUtils.NextHighestPowerOf2(a.Cols);
				CudaSigmaDiffDataBuffer<float> maxBuffer = _InternalInternalise(CreateDataBuffer(CreateUninitialisedArray(a.Rows)));
				CudaSigmaDiffDataBuffer<float> maxIndicesBuffer = _InternalInternalise(CreateDataBuffer(CreateUninitialisedArray(a.Rows)));
				CudaSigmaDiffDataBuffer<float> sumBuffer = _InternalInternalise(CreateDataBuffer(CreateUninitialisedArray(a.Rows)));

				int rowsPerBlock = ThreadsPerBlock / a.Cols;
				int elementsPerBlock = rowsPerBlock * a.Cols;
				int numBlocks = (len + elementsPerBlock - 1) / elementsPerBlock;

				RunKernel("Softmax_Rowwise_M", numBlocks * ThreadsPerBlock, ThreadsPerBlock * sizeof(float) * 2, aData.GetContextPointer(), maxBuffer.GetContextPointer(),
					maxIndicesBuffer.GetContextPointer(), sumBuffer.GetContextPointer(), a.Rows, a.Cols, colsNextPowerOf2, rData.GetContextPointer(), len);

				maxBuffer.FlagDeviceModified();
				maxIndicesBuffer.FlagDeviceModified();
				sumBuffer.FlagDeviceModified();

				op.AttachInfo("prevMaxs", maxBuffer);
				op.AttachInfo("prevMaxIndices", maxIndicesBuffer);
				op.AttachInfo("prevSums", sumBuffer);
			}

			rData.FlagDeviceModified();

			return result;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> CustomOp_DM_Backward(ShapedDataBufferView<float> origin,
			ShapedDataBufferView<float> adjoint, ShapedDataBufferView<float> primal, object customInfo)
		{
			if (!(customInfo is CustomOpHandle))
			{
				throw new InvalidOperationException($"Cannot invoke {nameof(CustomOp_DM_Forward)} with invalid custom info of type {customInfo.GetType()} (must be of type {nameof(CustomOpHandle)}).");
			}

			CustomOpHandle op = (CustomOpHandle)customInfo;

			if (!Enum.IsDefined(typeof(CustomOpType), op.Type))
			{
				throw new NotImplementedException($"Custom op {op} is not supported in {nameof(CustomOp_DM_Backward)}.");
			}

			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(CreateDataBuffer(CreateUninitialisedArray(origin.Length)));
			int len = (int)rData.Length;

			if (op.Type == CustomOpType.RowWiseSoftmax)
			{
				CudaSigmaDiffDataBuffer<float> originData = _InternalInternalise(origin);
				CudaSigmaDiffDataBuffer<float> adjointData = _InternalInternalise(adjoint);
				CudaSigmaDiffDataBuffer<float> primalData = _InternalInternalise(primal);
				CudaSigmaDiffDataBuffer<float> maxBuffer = op.GetInfo<CudaSigmaDiffDataBuffer<float>>("prevMaxs");
				CudaSigmaDiffDataBuffer<float> maxIndicesBuffer = op.GetInfo<CudaSigmaDiffDataBuffer<float>>("prevMaxIndices");
				CudaSigmaDiffDataBuffer<float> sumBuffer = op.GetInfo<CudaSigmaDiffDataBuffer<float>>("prevSums");

				RunKernel("Softmax_Rowwise_M_Backward", len, ThreadsPerBlock * sizeof(float) * 5, originData.GetContextPointer(), adjointData.GetContextPointer(),
					primalData.GetContextPointer(), maxBuffer.GetContextPointer(), maxIndicesBuffer.GetContextPointer(),
					sumBuffer.GetContextPointer(), rData.GetContextPointer(), origin.Rows, origin.Cols, ArrayUtils.NextHighestPowerOf2(origin.Cols), len);
			}

			rData.FlagDeviceModified();

			return new ShapedDataBufferView<float>(rData, origin.Rows, origin.Cols);
		}

		/// <inheritdoc />
		public void FillWithProbabilityMask(ISigmaDiffDataBuffer<float> a, double probability)
		{
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			int len = (int)aData.Length;

			RunKernel("FillWithProbabilityMask_V", len, aData.GetContextPointer(), (float)probability, len);

			aData.FlagDeviceModified();
		}

		/// <inheritdoc />
		public override float Mul_Dot_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> n)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override float L1Norm_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override float L2Norm_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override float SupNorm_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override unsafe float Sum_V(ISigmaDiffDataBuffer<float> a)
		{
			int len = a.Length, lenPartials = (len + ThreadsPerBlock - 1) / ThreadsPerBlock;

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> partialSums = (CudaSigmaDiffDataBuffer<float>)CreateDataBuffer(CreateUninitialisedArray(lenPartials));

			RunKernel("Sum_V", len, (uint)(ThreadsPerBlock * sizeof(float)), aData.GetContextPointer(), partialSums.GetContextPointer(), len);

			if (lenPartials > 1)
			{
				RunKernel("Sum_V", lenPartials, (uint)(ThreadsPerBlock * sizeof(float)), partialSums.GetContextPointer(), partialSums.GetContextPointer(), lenPartials);
			}

			partialSums.FlagDeviceModified();

			return partialSums.Data[0]; // TODO this is sub-optimal as we lose the advantages of having asynchronous GPU execution when explictly awaiting a result on host (e.g. the sum)
		}

		/// <inheritdoc />
		public override float Sum_M(ISigmaDiffDataBuffer<float> value)
		{
			return Sum_V(value);
		}

		/// <inheritdoc />
		public override unsafe int MaxIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(value);

			// TODO optimise using custom kernel for relative minimum value (not minimum magnitude like the (cu)blas implementation)
			int maxIndex = 0, len = (int)aData.Length;
			float maxValue = float.MinValue;

			fixed (float* aref = &aData.Data[aData.Offset])
			{
				for (int k = 0; k < len; k++)
				{
					if (aref[k] > maxValue)
					{
						maxValue = aref[k];
						maxIndex = k;
					}
				}
			}

			return maxIndex;
		}

		/// <inheritdoc />
		public override unsafe int MinIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(value);

			// TODO optimise using custom kernel for relative minimum value (not minimum magnitude like the (cu)blas implementation)
			int minIndex = 0, len = (int)aData.Length;
			float minValue = float.MaxValue;

			fixed (float* aref = &aData.Data[aData.Offset])
			{
				for (int k = 0; k < len; k++)
				{
					if (aref[k] < minValue)
					{
						minValue = aref[k];
						minIndex = k;
					}
				}
			}

			return minIndex;
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Add_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length == 0) return b.DeepCopy();
			if (b.Length == 0) return a.DeepCopy();

			b = b.DeepCopy();

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.GetContextBuffer(), 1, bData.GetContextBuffer(), 1);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override unsafe ISigmaDiffDataBuffer<float> Add_V_V_InPlace(ISigmaDiffDataBuffer<float> a, int aOffset, ISigmaDiffDataBuffer<float> b, int bOffset, int len)
		{
			if (len == 0)
			{
				return b;
			}

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			RunKernel("Add_V_V", len, aData.GetContextPointer(), aOffset, bData.GetContextPointer(), bOffset, len);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Add_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Sub_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Sub_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Sub_V_S(ISigmaDiffDataBuffer<float> a, float b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Mul_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Mul_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Mul_M_V_Add_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b, ISigmaDiffDataBuffer<float> obj2)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Mul_V_M(ISigmaDiffDataBuffer<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override FSharpOption<ISigmaDiffDataBuffer<float>> Solve_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override FSharpOption<ISigmaDiffDataBuffer<float>> SolveSymmetric_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Diagonal_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Map_F_V(MapOp mapOp, FSharpFunc<float, float> function, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Map_F_S_V(float other, MapOp mapOp, FSharpFunc<float, float> function, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Map2_F_V_V(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> function, ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Map_F_M(MapOp mapOp, FSharpFunc<float, float> f, ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			if (!_InternalOptimisedMap_F_M(mapOp, a, result))
			{
				int upper = a.DataBuffer.Offset + a.DataBuffer.Length;
				float[] aData = a.DataBuffer.Data, rData = result.DataBuffer.Data;

				for (int i = a.DataBuffer.Offset; i < upper; i++)
				{
					rData[i] = f.Invoke(aData[i]);
				}
			}
			else
			{
				_InternalInternalise(result).FlagDeviceModified();
			}

			return result;
		}

		private bool _InternalOptimisedMap_F_M(MapOp mapOp, ShapedDataBufferView<float> a, ShapedDataBufferView<float> result)
		{
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);
			int len = (int)aData.Length;

			if (mapOp.IsExp)
			{
				RunKernel("Exp_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}
			else if (mapOp.IsSqrt)
			{
				RunKernel("Sqrt_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}
			else if (mapOp.IsSign)
			{
				RunKernel("Sign_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}
			else if (mapOp.IsReL)
			{
				RunKernel("Rel_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}
			else if (mapOp.IsLog)
			{
				RunKernel("Log_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}
			else if (mapOp.IsSigmoid)
			{
				RunKernel("Sigmoid_V", len, aData.GetContextPointer(), rData.GetContextPointer(), len);

				return true;
			}

			return false;
		}


		/// <inheritdoc />
		public override ShapedDataBufferView<float> Map_F_S_M(float other, MapOp mapOp, FSharpFunc<float, float> function, ShapedDataBufferView<float> value)
		{
			if (_InternalOptimisedMapOp_F_S_M(other, mapOp, ref value))
			{
				return value;
			}

			return Map_F_M(mapOp, function, value);
		}

		private bool _InternalOptimisedMapOp_F_S_M(float other, MapOp mapOp, ref ShapedDataBufferView<float> a)
		{
			int len = a.Length;

			if (mapOp.IsDiv)
			{
				ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());
				CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
				CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);

				RunKernel("Div_S_V", len, other, aData.GetContextPointer(), rData.GetContextPointer(), len);

				a = result;
				rData.FlagDeviceModified();

				return true;
			}

			return false;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Map2_F_M_M(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> f, ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			if (_InternalOptimisedMapOp_F_M_M(mapOp, a, ref b))
			{
				return b;
			}

			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			float[] aData = a.DataBuffer.Data, bData = b.DataBuffer.Data, rData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, bOffset = b.DataBuffer.Offset, rOffset = result.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* bref = &bData[bOffset])
			fixed (float* resref = &rData[rOffset])
			{
				for (int i = 0; i < a.Length; i++)
				{
					resref[i] = f.Invoke(aref[i]).Invoke(bref[i]);
				}
			}

			return result;
		}

		private bool _InternalOptimisedMapOp_F_M_M(MapOp mapOp, ShapedDataBufferView<float> a, ref ShapedDataBufferView<float> b)
		{
			int len = b.Length;

			if (mapOp.IsDiv)
			{
				ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

				CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
				CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);
				CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);

				RunKernel("Div_V_V", len, aData.GetContextPointer(), bData.GetContextPointer(), rData.GetContextPointer(), len);

				b = result;

				return true;
			}

			return false;
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> ReshapeCopy_MRows_V(ShapedDataBufferView<float> value)
		{
			if (value.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			return value.DataBuffer.DeepCopy();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_Out_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0) return b.DeepCopy();
			if (b.Length == 0) return a.DeepCopy();

			b = b.DeepCopy();

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.GetContextBuffer(), 1, bData.GetContextBuffer(), 1);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_M_M_InPlace(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0) return b;
			if (b.Length == 0) return a;

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.GetContextBuffer(), 1, bData.GetContextBuffer(), 1);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Add_S_M(float other, ShapedDataBufferView<float> a)
		{
			int len = a.Length;

			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);

			RunKernel("Add_V_S", len, aData.GetContextPointer(), other, rData.GetContextPointer(), len);

			rData.FlagDeviceModified();

			return result;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_V_MCols(ISigmaDiffDataBuffer<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Sub_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0) return b.DeepCopy();
			if (b.Length == 0) return a.DeepCopy();

			a = a.DeepCopy();

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			float alpha = -1.0f;

			CudaBlasHandle.Axpy(alpha, bData.GetContextBuffer(), 1, aData.GetContextBuffer(), 1);

			aData.FlagDeviceModified();

			return a;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Sub_M_S(ShapedDataBufferView<float> a, float b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			int len = (int)aData.Length;

			RunKernel("Sub_V_S", len, aData.GetContextPointer(), b, rData.GetContextPointer(), len);

			rData.FlagDeviceModified();

			return result;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Sub_S_M(float other, ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			int len = (int)aData.Length;

			RunKernel("Sub_S_V", len, other, aData.GetContextPointer(), rData.GetContextPointer(), len);

			rData.FlagDeviceModified();

			return result;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length * b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);
			CudaSigmaDiffDataBuffer<float> zData = (CudaSigmaDiffDataBuffer<float>)CreateDataBuffer(CreateUninitialisedArray(a.Rows * b.Cols));

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = b.Cols, k = b.Rows;

			CudaBlasHandle.Gemm(Operation.NonTranspose, Operation.NonTranspose, n, m, k, alpha, bData.GetContextBuffer(), n,
				aData.GetContextBuffer(), k, beta, zData.GetContextBuffer(), n);

			zData.FlagDeviceModified();

			return new ShapedDataBufferView<float>(zData, a.Rows, b.Cols);
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_S_M(float a, ShapedDataBufferView<float> b)
		{
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();

			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			CudaBlasHandle.Scale(a, bData.GetContextBuffer(), 1);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_M_M_Add_V_MCols(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b, ISigmaDiffDataBuffer<float> obj2)
		{
			throw new NotImplementedException();
		}

		public override ISigmaDiffDataBuffer<float> Add_M_Colwise_V_InPlace(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			long[] transposedShape = new long[a.Shape.Length];

			for (int i = 0; i < transposedShape.Length; i++)
			{
				transposedShape[i] = a.Shape[a.Shape.Length - 1 - i];
			}

			ShapedDataBufferView<float> transposed = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), transposedShape);
			CudaSigmaDiffDataBuffer<float> tData = _InternalInternalise(transposed);

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = a.Cols;

			CudaBlasHandle.Geam(Operation.Transpose, Operation.NonTranspose, m, n, alpha, aData.GetContextBuffer(), n, tData.GetContextBuffer(), m, beta, tData.GetContextBuffer(), m);

			tData.FlagDeviceModified();

			int len = (int)aData.Length;

			int rowsPerBlock = ThreadsPerBlock / a.Rows;
			int elementsPerBlock = rowsPerBlock * a.Rows;
			int numBlocks = (len + elementsPerBlock - 1) / elementsPerBlock;
			CudaSigmaDiffDataBuffer<float> sumBuffer = (CudaSigmaDiffDataBuffer<float>)CreateDataBuffer(CreateUninitialisedArray(numBlocks * rowsPerBlock));

			RunKernel("Sum_M_Rowwise", numBlocks * ThreadsPerBlock, (uint)ThreadsPerBlock * sizeof(float), tData.GetContextPointer(), a.Cols, a.Rows,
				ArrayUtils.NextHighestPowerOf2(a.Rows), sumBuffer.GetContextPointer(), len);

			int sumLen = (int)sumBuffer.Length;
			int sumRowsPerBlock = ThreadsPerBlock / rowsPerBlock;
			int sumElementsPerBlock = sumRowsPerBlock * rowsPerBlock;
			int sumNumBlocks = (sumLen + sumElementsPerBlock - 1) / sumElementsPerBlock;

			RunKernel("Add_M_Rowwise_V_InPlace", sumNumBlocks * ThreadsPerBlock, (uint)ThreadsPerBlock * sizeof(float), sumBuffer.GetContextPointer(), numBlocks,
				rowsPerBlock, ArrayUtils.NextHighestPowerOf2(rowsPerBlock), bData.GetContextPointer(), sumLen);

			bData.FlagDeviceModified();

			return b;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Mul_Had_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(CreateZeroArray(b.Length)), b.Shape);
			}
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(CreateZeroArray(a.Length)), a.Shape);
			}

			int len = Math.Min(a.Length, b.Length);
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(b.Length)), (long[])b.Shape.Clone());
			CudaSigmaDiffDataBuffer<float> rData = _InternalInternalise(result);
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			RunKernel("Mul_Had_V_V", len, aData.GetContextPointer(), bData.GetContextPointer(), rData.GetContextPointer(), len);

			rData.FlagDeviceModified();

			return result;
		}

		/// <inheritdoc />
		public override FSharpOption<ShapedDataBufferView<float>> Inverse_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override FSharpOption<float> Det_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Transpose_M(ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ShapedDataBufferView<float> transposed = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			for (int i = 0; i < transposed.Shape.Length; i++)
			{
				transposed.Shape[i] = a.Shape[a.Shape.Length - 1 - i];
			}

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> tData = _InternalInternalise(transposed);

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = a.Cols;

			CudaBlasHandle.Geam(Operation.Transpose, Operation.NonTranspose, m, n, alpha, aData.GetContextBuffer(), n, aData.GetContextBuffer(), m, beta, tData.GetContextBuffer(), m);

			tData.FlagDeviceModified();

			return transposed;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Permute_M(ShapedDataBufferView<float> a, int[] rearrangedDimensions)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			int rank = a.Shape.Length, len = a.Length;

			float[] rearrangedDimensionsFloat = CreateUninitialisedArray(rank);

			for (int i = 0; i < rearrangedDimensionsFloat.Length; i++)
			{
				rearrangedDimensionsFloat[i] = rearrangedDimensions[i];
			}

			ShapedDataBufferView<float> permuted = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), ArrayUtils.PermuteArray(a.Shape, rearrangedDimensions));

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> pData = _InternalInternalise(permuted);

			CudaSigmaDiffDataBuffer<float> permutedDimensions = _InternalInternalise(CreateDataBuffer(rearrangedDimensionsFloat));
			CudaSigmaDiffDataBuffer<float> originalStrides = _InternalInternalise(CreateDataBuffer(GetFloatStrides(a.Shape)));
			CudaSigmaDiffDataBuffer<float> permutedStrides = _InternalInternalise(CreateDataBuffer(GetFloatStrides(permuted.Shape)));

			RunKernel("Permute_M", len, (uint)rank * 2 * ThreadsPerBlock * sizeof(float), aData.GetContextPointer(), permutedDimensions.GetContextPointer(), originalStrides.GetContextPointer(),
				pData.GetContextPointer(), permutedStrides.GetContextPointer(), rank, len);

			pData.FlagDeviceModified();

			return permuted;
		}

		private float[] GetFloatStrides(long[] shape)
		{
			float[] strides = new float[shape.Length];

			long currentStride = 1;
			for (int i = shape.Length - 1; i >= 0; i--)
			{
				strides[i] = currentStride;
				currentStride *= shape[i];
			}

			return strides;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Reshape_M(ShapedDataBufferView<float> array, long[] newShape)
		{
			ShapedDataBufferView<float> reshaped = new ShapedDataBufferView<float>(array.DataBuffer, newShape);

			return reshaped;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			int n = value.Length / rows;

			return new ShapedDataBufferView<float>(value.DeepCopy(), rows, n);
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> row)
		{
			if (row.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			int rowLength = row.Length;
			float[] result = CreateUninitialisedArray(rows * rowLength);
			CudaSigmaDiffDataBuffer<float> rowSubData = _InternalInternalise(row);
			CudaSigmaDiffDataBuffer<float> resultData = (CudaSigmaDiffDataBuffer<float>)CreateDataBuffer(result);

			if (!rowSubData.IsInitialisedInContext())
			{
				float[] rowData = row.Data;
				int sourceOffset = row.Offset;
				int destinationOffset = 0;

				for (int i = 0; i < rows; i++)
				{
					Buffer.BlockCopy(rowData, sourceOffset * sizeof(float), result, destinationOffset * sizeof(float), rowLength * sizeof(float));

					destinationOffset += rowLength;
				}
			}
			else
			{
				resultData.InitialiseCudaBuffer(copyHostToDevice: false);

				RunKernel("RepeatReshapeCopy_V_MRows", rowLength, rowSubData.GetContextPointer(), resultData.GetContextPointer(), rows, rowLength, result.Length);

				resultData.FlagDeviceModified();
			}

			return new ShapedDataBufferView<float>(resultData, rows, rowLength);
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		~CudaFloat32BackendHandle()
		{
			// forcibly dispose all device buffers, not sure if this is such a great idea
			foreach (CudaDeviceVariable<float> deviceBuffer in _allocatedDeviceBuffers.Values)
			{
				deviceBuffer.Dispose();
			}
		}
	}
}
