/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics;
using DiffSharp.Backend;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using Microsoft.FSharp.Core;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaFloat32BackendHandle : DiffSharpBackendHandle<float>
	{
		internal readonly CudaBlas CudaBlasHandle;
		internal readonly CudaContext CudaContext;

		public CudaFloat32BackendHandle(int deviceId, long backendTag) : base(backendTag)
		{
			CudaBlasHandle = new CudaBlas();
			CudaContext = new CudaContext(deviceId);
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

			// TODO check in CudaBuffer if there is host access, otherwise keep in device all the time until it is accessed
			//  (i.e. reduce host <-> device transfers by a lot for better performance)
			aData.CopyFromHostToDevice();
			bData.CopyFromHostToDevice();

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.CudaBuffer, 1, bData.CudaBuffer, 1);

			bData.CopyFromDeviceToHost();

			return b;
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Add_V_V_InPlace(ISigmaDiffDataBuffer<float> obj0, int obj1, ISigmaDiffDataBuffer<float> obj2, int obj3, int obj4)
		{
			throw new NotImplementedException();
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
		public override ShapedDataBufferView<float> Map2_F_M_M(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> function, ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
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

			// TODO check in CudaBuffer if there is host access, otherwise keep in device all the time until it is accessed
			//  (i.e. reduce host <-> device transfers by a lot for better performance)
			aData.CopyFromHostToDevice();
			bData.CopyFromHostToDevice();

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.CudaBuffer, 1, bData.CudaBuffer, 1);

			bData.CopyFromDeviceToHost();

			return b;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_M_M_InPlace(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0) return b;
			if (b.Length == 0) return a;

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> bData = _InternalInternalise(b);

			// TODO check in CudaBuffer if there is host access, otherwise keep in device all the time until it is accessed
			//  (i.e. reduce host <-> device transfers by a lot for better performance)
			aData.CopyFromHostToDevice();
			bData.CopyFromHostToDevice();

			float alpha = 1.0f;

			CudaBlasHandle.Axpy(alpha, aData.CudaBuffer, 1, bData.CudaBuffer, 1);

			bData.CopyFromDeviceToHost();

			return b;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_S_M(float a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
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

			aData.CopyFromHostToDevice();
			bData.CopyFromHostToDevice();

			float alpha = -1.0f;

			CudaBlasHandle.Axpy(alpha, bData.CudaBuffer, 1, aData.CudaBuffer, 1);

			aData.CopyFromDeviceToHost();

			return a;
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Sub_M_S(ShapedDataBufferView<float> a, float b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			a = a.DeepCopy();
			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);

			aData.CopyFromHostToDevice();

			CudaBlasHandle.Axpy(1.0f, b, 0, aData.CudaBuffer, 1);

			aData.CopyFromDeviceToHost();

			return a;
		}

		/// <inheritdoc />
		public override unsafe ShapedDataBufferView<float> Sub_S_M(float other, ShapedDataBufferView<float> a)
		{
			int len = a.Length;
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), (long[]) a.Shape.Clone());
			float[] aData = a.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			// TODO optimise using custom kernel for subtracting array from constant (doesn't work with blas incx trick because then the result is inaccessible)
			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				for (int i = 0; i < len; i++)
				{
					resref[i] = other - aref[i];
				}
			}

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

			aData.CopyFromHostToDevice();
			bData.CopyFromHostToDevice();

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = b.Cols, k = b.Rows;

			CudaBlasHandle.Gemm(Operation.NonTranspose, Operation.NonTranspose, n, m, k, alpha, bData.CudaBuffer, n, aData.CudaBuffer, k, beta, zData.CudaBuffer, n);

			zData.CopyFromDeviceToHost();

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

			bData.CopyFromHostToDevice();

			CudaBlasHandle.Scale(a, bData.CudaBuffer, 1);

			bData.CopyFromDeviceToHost();

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
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), (long[]) b.Shape.Clone());

			float[] aData = a.DataBuffer.Data, bData = b.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, bOffset = b.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			// TODO optimise using custom kernel for hadamard product (no support in cublas either)
			fixed (float* aref = &aData[aOffset])
			fixed (float* bref = &bData[bOffset])
			fixed (float* resref = &resData[resOffset])
			{
				for (int i = 0; i < len; ++i)
				{
					resref[i] = aref[i] * bref[i];
				}
			}

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

			ShapedDataBufferView<float> transposed = a.DeepCopy();

			for (int i = 0; i < transposed.Shape.Length; i++)
			{
				transposed.Shape[i] = a.Shape[a.Shape.Length - 1 - i];
			}

			CudaSigmaDiffDataBuffer<float> aData = _InternalInternalise(a);
			CudaSigmaDiffDataBuffer<float> tData = _InternalInternalise(transposed);

			float alpha = 1.0f, beta = 0.0f;
			int m = a.Rows, n = a.Cols;

			aData.CopyFromHostToDevice();
			tData.CopyFromHostToDevice();

			CudaBlasHandle.Geam(Operation.Transpose, Operation.NonTranspose, m, n, alpha, aData.CudaBuffer, m, tData.CudaBuffer, m, beta, tData.CudaBuffer, m);

			tData.CopyFromDeviceToHost();

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
