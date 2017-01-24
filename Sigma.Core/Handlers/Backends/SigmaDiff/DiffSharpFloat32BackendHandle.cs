/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using Microsoft.FSharp.Core;
using Sigma.Core.Utils;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	/// <summary>
	/// A DiffSharp backend handle for 32-bit floats as passed to the underlying DiffSharp implementation.
	/// </summary>
	public unsafe class DiffSharpFloat32BackendHandle : DiffSharpBackendHandle<float>
	{
		/// <summary>
		/// Create a DiffSharpFloat32BackendHandle with a certain BLAS and LAPACK backend and an associated handle tag. 
		/// </summary>
		/// <param name="blasBackend"></param>
		/// <param name="lapackBackend"></param>
		/// <param name="backendTag"></param>
		public DiffSharpFloat32BackendHandle(IBlasBackend blasBackend, ILapackBackend lapackBackend, long backendTag) : base(blasBackend, lapackBackend, backendTag)
		{
		}

		public override ISigmaDiffDataBuffer<float> CreateDataBuffer(float[] values)
		{
			return new SigmaDiffDataBuffer<float>(values, backendTag: BackendTag);
		}

		public override float L1Norm_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0.0f;
			}

			fixed (float* aref = &value.Data[value.Offset])
			{
				int len = value.Length;
				int inca = 1;

				return BlasBackend.Sasum(&len, aref, &inca);
			}
		}

		public override float L2Norm_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0.0f;
			}

			fixed (float* aref = &value.Data[value.Offset])
			{
				int len = value.Length;
				int inca = 1;

				return BlasBackend.Snrm2(&len, aref, &inca);
			}
		}

		public override float SupNorm_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0.0f;
			}

			fixed (float* aref = &value.Data[value.Offset])
			{
				int len = value.Length;
				int inca = 1;

				int i = BlasBackend.Isamax(&len, aref, &inca);

				return value.Data[value.Offset + i - 1];
			}
		}

		public override float Sum_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0.0f;
			}

			float sum = 0.0f;

			int upper = value.Offset + value.Length;
			for (int i = value.Offset; i < upper; i++)
			{
				sum += value.Data[i];
			}

			return sum;
		}

		public override float Sum_M(ISigmaDiffDataBuffer<float> value)
		{
			return Sum_V(value);
		}

		public override ISigmaDiffDataBuffer<float> Add_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length == 0)
			{
				return b.DeepCopy();
			}
			if (b.Length == 0)
			{
				return a.DeepCopy();
			}

			b = b.DeepCopy();
			fixed (float* aref = &a.Data[a.Offset])
			fixed (float* bref = &b.Data[b.Offset])
			{
				int len = Math.Min(a.Length, b.Length);
				int inca = 1, incb = 1;
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, bref, &incb);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Add_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			if (b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			b = b.DeepCopy();
			fixed (float* bref = &b.Data[b.Offset])
			{
				int len = Math.Min(1, b.Length);
				int inca = 0, incb = 1;
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, &a, &inca, bref, &incb);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Sub_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length == 0)
			{
				return b.DeepCopy();
			}
			if (b.Length == 0)
			{
				return a.DeepCopy();
			}

			b = b.DeepCopy();
			fixed (float* aref = &a.Data[a.Offset])
			fixed (float* bref = &b.Data[b.Offset])
			{
				int len = Math.Min(a.Length, b.Length);
				int inca = 1, incb = 1;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, bref, &incb, aref, &inca);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Sub_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			if (b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			b = b.DeepCopy();
			fixed (float* bref = &b.Data[b.Offset])
			{
				int len = b.Length;
				int inca = 0, incb = 1;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, &a, &inca, bref, &incb);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Sub_V_S(ISigmaDiffDataBuffer<float> a, float b)
		{
			if (a.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			a = a.DeepCopy();
			fixed (float* aref = &a.Data[a.Offset])
			{
				int len = a.Length;
				int inca = 1, incb = 0;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, &b, &incb);
			}

			return a;
		}

		public override ISigmaDiffDataBuffer<float> Mul_S_V(float a, ISigmaDiffDataBuffer<float> b)
		{
			if (b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			b = b.DeepCopy();
			fixed (float* bref = &b.Data[b.Offset])
			{
				int len = b.Length;
				int incx = 1;

				BlasBackend.Sscal(&len, &a, bref, &incx);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Mul_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length * b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			ISigmaDiffDataBuffer<float> z = CreateDataBuffer(new float[a.Rows]);

			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &b.Data[b.Offset])
			fixed (float* zref = &z.Data[z.Offset])
			{
				char trans = 'T';
				int m = a.Cols, n = a.Rows;
				int incb = 1, incz = 1;
				float alpha = 1.0f, beta = 0.0f;

				BlasBackend.Sgemv(&trans, &m, &n, &alpha, aref, &m, bref, &incb, &beta, zref, &incz);
			}

			return z;
		}

		public override ISigmaDiffDataBuffer<float> Mul_M_V_Add_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b, ISigmaDiffDataBuffer<float> obj2)
		{
			throw new NotImplementedException();
		}

		public override float Mul_Dot_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> n)
		{
			throw new NotImplementedException();
		}

		public override ISigmaDiffDataBuffer<float> Mul_V_M(ISigmaDiffDataBuffer<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length * b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			ISigmaDiffDataBuffer<float> z = CreateDataBuffer(new float[b.Rows]);

			fixed (float* aref = &a.Data[a.Offset])
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			fixed (float* zref = &z.Data[z.Offset])
			{
				char trans = 'T';
				int m = b.Cols, n = b.Rows;
				int incb = 1, incz = 1;
				float alpha = 1.0f, beta = 0.0f;

				BlasBackend.Sgemv(&trans, &m, &n, &alpha, aref, &m, bref, &incb, &beta, zref, &incz);
			}

			return z;
		}

		public override FSharpOption<ISigmaDiffDataBuffer<float>> Solve_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		public override FSharpOption<ISigmaDiffDataBuffer<float>> SolveSymmetric_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		public override ISigmaDiffDataBuffer<float> Diagonal_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		public override ISigmaDiffDataBuffer<float> Map_F_V(FSharpFunc<float, float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (b.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			b = b.DeepCopy();

			int upper = b.Offset + b.Length;
			for (int i = b.Offset; i < upper; i++)
			{
				b.Data[i] = a.Invoke(b.Data[i]);
			}

			return b;
		}

		public override ISigmaDiffDataBuffer<float> Map2_F_V_V(FSharpFunc<float, FSharpFunc<float, float>> f, ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length == 0)
			{
				return Map2_F_V_V(f, CreateDataBuffer(new float[b.Length]), b);
			}
			if (b.Length == 0)
			{
				return Map2_F_V_V(f, a, CreateDataBuffer(new float[a.Length]));
			}

			b = b.DeepCopy();

			for (int i = 0; i < a.Length; i++)
			{
				b.Data[i] = f.Invoke(a.Data[i + a.Offset]).Invoke(b.Data[i + b.Offset]);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Mul_Out_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length * b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ISigmaDiffDataBuffer<float> z = CreateDataBuffer(new float[a.Length * b.Length]);
			int m = b.Length, n = a.Length;

			fixed (float* aref = &a.Data[a.Offset])
			fixed (float* bref = &b.Data[b.Offset])
			fixed (float* zref = &z.Data[z.Offset])
			{
				int inca = 1, incb = 1;

				float alpha = 1.0f;

				BlasBackend.Sger(&m, &n, &alpha, aref, &inca, bref, &incb, zref, &m);
			}

			return new ShapedDataBufferView<float>(z, m, n);
		}

		public override ShapedDataBufferView<float> Add_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return b.DeepCopy();
			}
			if (b.Length == 0)
			{
				return a.DeepCopy();
			}

			b = b.DeepCopy();
			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = Math.Min(a.Length, b.Length);
				int inca = 1, incb = 1;
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, bref, &incb);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Add_S_M(float a, ShapedDataBufferView<float> b)
		{
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();
			fixed (float* aref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = b.Length;
				int inca = 1, incb = 0;
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, &a, &incb);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Add_V_MCols(ISigmaDiffDataBuffer<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		public override ShapedDataBufferView<float> Sub_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return b.DeepCopy();
			}
			if (b.Length == 0)
			{
				return a.DeepCopy();
			}

			b = b.DeepCopy();
			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = Math.Min(a.Length, b.Length);
				int inca = 1, incb = 1;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, bref, &incb);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Sub_M_S(ShapedDataBufferView<float> a, float b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			a = a.DeepCopy();
			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			{
				int len = a.Length;
				int inca = 1, incb = 0;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, &b, &incb, aref, &inca);
			}

			return a;
		}

		public override ShapedDataBufferView<float> Sub_S_M(float a, ShapedDataBufferView<float> b)
		{
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = b.Length;
				int inca = 0, incb = 1;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, &a, &inca, bref, &incb);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Mul_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length * b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ISigmaDiffDataBuffer<float> z = CreateDataBuffer(new float[a.Rows * b.Cols]);

			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			fixed (float* zref = &z.Data[z.Offset])
			{
				char transa = 'N', transb = 'N';
				float alpha = 1.0f, beta = 0.0f;
				int m = a.Rows, n = b.Cols, k = b.Rows;
				
				BlasBackend.Sgemm(&transa, &transb, &n, &m, &k, &alpha, bref, &n, aref, &k, &beta, zref, &n);
			}

			return new ShapedDataBufferView<float>(z, a.Rows, b.Cols);
		}

		public override ShapedDataBufferView<float> Mul_S_M(float a, ShapedDataBufferView<float> b)
		{
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = b.Length;
				int incx = 1;

				BlasBackend.Sscal(&len, &a, bref, &incx);
			}

			return b;
		}

		public override ShapedDataBufferView<float> Mul_M_M_Add_V_MCols(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b, ISigmaDiffDataBuffer<float> c)
		{
			throw new NotImplementedException();
		}

		public override ShapedDataBufferView<float> Mul_Had_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[b.Length]), b.Shape);
			}
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[a.Length]), a.Shape);
			}

			//TODO update with BLAS hadamard implementation
			b = b.DeepCopy();
			int len = Math.Min(a.Length, b.Length);
			int offsetA = a.DataBuffer.Offset, offsetB = b.DataBuffer.Offset;
			float[] dataA = a.DataBuffer.Data;
			float[] dataB = b.DataBuffer.Data;
			for (int i = 0; i < len; i++)
			{
				dataB[i + offsetB] = dataA[i + offsetA] * dataB[i + offsetB];
			}

			return b;
		}

		public override FSharpOption<ShapedDataBufferView<float>> Inverse_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		public override FSharpOption<float> Det_M(ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return FSharpOption<float>.Some(0.0f);
			}

			a = a.DeepCopy();

			int info = 0;
			int[] ipiv = new int[Math.Min(a.Rows, a.Cols)];

			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (int* ipivref = &ipiv[0])
			{
				int m = a.Rows, n = a.Cols;

				LapackBackend.Sgetrf_(&m, &n, aref, &m, ipivref, &info);
			}

			if (info != 0)
			{
				return FSharpOption<float>.None;
			}

			float det = 1.0f;

			for (int i = 0; i < ipiv.Length; i++)
			{
				det *= ipiv[i] != i + 1 ? -a[i, i] : a[i, i];
			}

			return FSharpOption<float>.Some(det);
		}

		public override ShapedDataBufferView<float> Transpose_M(ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			ShapedDataBufferView<float> transposed = a.DeepCopy();

			for (var i = 0; i < transposed.Shape.Length; i++)
			{
				transposed.Shape[i] = a.Shape[a.Shape.Length - 1 - i];
			}

			return transposed;
		}

		public override ShapedDataBufferView<float> Map_F_M(FSharpFunc<float, float> f, ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			a = a.DeepCopy();

			int upper = a.DataBuffer.Offset + a.DataBuffer.Length;
			for (int i = a.DataBuffer.Offset; i < upper; i++)
			{
				a.DataBuffer.Data[i] = f.Invoke(a.DataBuffer.Data[i]);
			}

			return a;
		}

		public override ShapedDataBufferView<float> Map2_F_M_M(FSharpFunc<float, FSharpFunc<float, float>> f, ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();

			for (int i = 0; i < a.Length; i++)
			{
				b.DataBuffer.Data[i] = f.Invoke(a.DataBuffer.Data[i + a.DataBuffer.Offset]).Invoke(b.DataBuffer.Data[i + b.DataBuffer.Offset]);
			}

			return a;
		}

		public override ISigmaDiffDataBuffer<float> ReshapeCopy_MRows_V(ShapedDataBufferView<float> value)
		{
			throw new NotImplementedException();
		}

		public override ShapedDataBufferView<float> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}
	}
}
