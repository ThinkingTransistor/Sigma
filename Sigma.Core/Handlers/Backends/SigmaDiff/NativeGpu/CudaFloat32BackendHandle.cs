/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using DiffSharp.Backend;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using ManagedCuda.VectorTypes;
using Microsoft.FSharp.Core;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaFloat32BackendHandle : DiffSharpBackendHandle<float>
	{
		internal CudaBlas CudaBlasHandle;
		internal readonly CudaContext CudaContext;
		internal ConditionalWeakTable<object, CudaDeviceVariable<float>> _allocatedDeviceBuffers;

		private const int ThreadsPerBlock = 256;
		private CUmodule _kernelModule;
		private IDictionary<string, CudaKernel> _loadedKernels;

		public CudaFloat32BackendHandle(int deviceId, long backendTag) : base(backendTag)
		{
			CudaContext = new CudaContext(deviceId);

			_kernelModule = CudaContext.LoadModulePTX("Dependencies/sigmakernels.ptx");
			_loadedKernels = LoadKernels(_kernelModule);

			_allocatedDeviceBuffers = new ConditionalWeakTable<object, CudaDeviceVariable<float>>();

			BindToContext();
		}

		private IDictionary<string, CudaKernel> LoadKernels(CUmodule kernelModule)
		{
			IDictionary<string, CudaKernel> loadedKernels = new Dictionary<string, CudaKernel>();

			loadedKernels.Add("Sub_V_S", new CudaKernel("_Z7Sub_V_SPffi", kernelModule, CudaContext));
			loadedKernels.Add("Sub_S_V", new CudaKernel("_Z7Sub_S_VfPfi", kernelModule, CudaContext));
			loadedKernels.Add("Add_V_S", new CudaKernel("_Z7Add_V_SPffi", kernelModule, CudaContext));
			loadedKernels.Add("Mul_Had_V_V", new CudaKernel("_Z11Mul_Had_V_VPKfPfi", kernelModule, CudaContext));

			return loadedKernels;
		}

		private void RunKernel(string kernelName, int elementCount, params object[] kernelParameters)
		{
			if (!_loadedKernels.ContainsKey(kernelName))
			{
				throw new InvalidOperationException($"Unable to run kernel, kernel with name {kernelName} is not loaded.");
			}

			CudaKernel kernel = _loadedKernels[kernelName];

			kernel.BlockDimensions = ThreadsPerBlock;
			kernel.GridDimensions = (elementCount + ThreadsPerBlock - 1) / ThreadsPerBlock;

			kernel.Run(kernelParameters);
		}

		internal void BindToContext()
		{
			CudaContext.SetCurrent();
			CudaBlasHandle = new CudaBlas();
		}

		/// <summary>
		/// Allocate a CUDA buffer on the used device for a certain host array.
		/// </summary>
		/// <typeparam name="T">The buffer type (only float32 supported here).</typeparam>
		/// <param name="hostData">The host version this data.</param>
		/// <param name="cudaLengthBytes">The length in bytes as a SizeT struct (if allocation is required).</param>
		/// <returns>A CUDA buffer corresponding to the host array of the required size (cached if already exists, otherwise newly allocated).</returns>
		internal CudaDeviceVariable<T> AllocateDeviceBuffer<T>(T[] hostData, SizeT cudaLengthBytes) where T : struct
		{
			// TODO this casting and type checking is absolutely horribly, need to improve the way the data buffer accesses this so that it can be either truly dynamic or fixed type
			if (typeof(T) != typeof(float)) throw new InvalidOperationException($"{nameof(CudaFloat32BackendHandle)} can only allocate float32 device buffers, given type {typeof(T)} is not valid.");

			// The caching here works because I'm essentially tagging along with the system memory caching done in DiffSharpBackendHandle<T>.
			// Basically, the idea is that every host array has a corresponding device buffer, and because the host arrays are already reused as necessary,
			//  the device buffers are too as they are weakly associated with the host arrays in a weak table. This also automatically takes care of "freeing" device buffers.
			CudaDeviceVariable<float> deviceBuffer;
			if (_allocatedDeviceBuffers.TryGetValue(hostData, out deviceBuffer))
			{
				// TODO temp fix to make sure the returned data is of the right size, maybe manage offset / length separately for host data, unfortunately cuda doesn't support buffer overlap					 
				if (deviceBuffer.SizeInBytes == cudaLengthBytes)
				{
					return (CudaDeviceVariable<T>)(object)deviceBuffer;
				}
				else
				{
					_allocatedDeviceBuffers.Remove(hostData);
				}
			}

			deviceBuffer = new CudaDeviceVariable<float>(CudaContext.AllocateMemory(cudaLengthBytes), true, cudaLengthBytes);

			_allocatedDeviceBuffers.Add(hostData, deviceBuffer);

			return (CudaDeviceVariable<T>)(object)deviceBuffer;
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
			float[] aData = a.Data;
			int aOffset = a.Offset, len = a.Length;
			float result = 0.0f;

			// TODO optimise using custom kernel for relative sum (not using absolutes like in the cublas implementation)
			fixed (float* aref = &aData[aOffset])
			{
				for (int i = 0; i < len; ++i)
				{
					result += aref[i];
				}
			}

			return result;
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

			// TODO optimise using custom kernel for offset / limited vector addition (reallocating device pointer would be too cumbersome and slow)
			fixed (float* aref = &a.Data[a.Offset + aOffset])
			fixed (float* bref = &b.Data[b.Offset + bOffset])
			{
				for (int i = 0; i < len; i++)
				{
					bref[i] = aref[i] + bref[i];
				}
			}

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

			a = a.DeepCopy();

			int upper = a.DataBuffer.Offset + a.DataBuffer.Length;
			float[] data = a.DataBuffer.Data;

			for (int i = a.DataBuffer.Offset; i < upper; i++)
			{
				data[i] = f.Invoke(data[i]);
			}

			return a;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Map_F_S_M(float other, MapOp mapOp, FSharpFunc<float, float> function, ShapedDataBufferView<float> value)
		{
			return Map_F_M(mapOp, function, value);
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

			b = b.DeepCopy();

			float[] aData = a.DataBuffer.Data, bData = b.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, bOffset = b.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* bref = &bData[bOffset])
			{
				for (int i = 0; i < a.Length; i++)
				{
					bref[i] = f.Invoke(aref[i]).Invoke(bData[i]);
				}
			}

			return b;
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

			a = a.DeepCopy();

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			RunKernel("Add_V_S", len, aData.GetContextBuffer().DevicePointer, other, len);

			aData.FlagDeviceModified();

			return a;
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

			a = a.DeepCopy();
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			int len = (int)aData.Length;

			RunKernel("Sub_V_S", len, aData.GetContextBuffer().DevicePointer, b, len);

			aData.FlagDeviceModified();

			return a;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Sub_S_M(float other, ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			a = a.DeepCopy();
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			int len = (int)aData.Length;

			RunKernel("Sub_S_V", len, other, aData.GetContextBuffer().DevicePointer, len);

			aData.FlagDeviceModified();

			return a;
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
			b = b.DeepCopy();

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			RunKernel("Mul_Had_V_V", len, aData.GetContextBuffer().DevicePointer, bData.GetContextBuffer().DevicePointer, len);

			bData.FlagDeviceModified();
			
			return b;
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

			ShapedDataBufferView<float> transposed = a.DeepCopy();

			for (int i = 0; i < transposed.Shape.Length; i++)
			{
				transposed.Shape[i] = a.Shape[a.Shape.Length - 1 - i];
			}

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> tData = _InternalInternalise(transposed);

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = a.Cols;

			CudaBlasHandle.Geam(Operation.Transpose, Operation.NonTranspose, m, n, alpha, aData.GetContextBuffer(), n, tData.GetContextBuffer(), m, beta, tData.GetContextBuffer(), m);

			tData.FlagDeviceModified();

			return transposed;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Permute_M(ShapedDataBufferView<float> array, int[] rearrangedDimensions)
		{
			throw new NotImplementedException();
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
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> row)
		{
			if (row.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			int rowLength = row.Length;
			float[] result = CreateUninitialisedArray(rows * rowLength);
			float[] rowData = row.Data;
			int sourceOffset = row.Offset;
			int destinationOffset = 0;

			for (int i = 0; i < rows; i++)
			{
				Buffer.BlockCopy(rowData, sourceOffset * sizeof(float), result, destinationOffset * sizeof(float), rowLength * sizeof(float));

				destinationOffset += rowLength;
			}

			return new ShapedDataBufferView<float>(CreateDataBuffer(result), rows, rowLength);
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}
	}
}
