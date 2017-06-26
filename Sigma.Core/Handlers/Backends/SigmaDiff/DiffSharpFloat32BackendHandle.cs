/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using DiffSharp;
using DiffSharp.Backend;
using Microsoft.FSharp.Core;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
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
		/// <param name="blasBackend">The BLAS backend to use (must use 32-bit floats).</param>
		/// <param name="lapackBackend">The LAPACK backend to use (must use 32-bit floats).</param>
		/// <param name="backendTag">The backend tag to use.</param>
		public DiffSharpFloat32BackendHandle(IBlasBackend blasBackend, ILapackBackend lapackBackend, long backendTag) : base(blasBackend, lapackBackend, backendTag)
		{
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.CreateDataBuffer"/>
		public override ISigmaDiffDataBuffer<float> CreateDataBuffer(float[] values)
		{
			return new SigmaDiffDataBuffer<float>(values, backendTag: BackendTag);
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.L1Norm_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.L2Norm_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.SupNorm_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sum_V"/>
		public override float Sum_V(ISigmaDiffDataBuffer<float> a)
		{
			if (a.Length == 0)
			{
				return 0.0f;
			}

			int simdLength = Vector<float>.Count, len = a.Length;
			float[] aData = a.Data, tempData = CreateZeroArray(simdLength);
			int aOffset = a.Offset;
			float result = 0.0f;

			fixed (float* aref = &aData[aOffset])
			{
				Vector<float> vt = new Vector<float>(tempData);

				int i;
				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					vt += va;
				}

				for (; i < len; ++i)
				{
					result += aref[i];
				}

				result += Vector.Dot<float>(vt, Vector<float>.One);
			}

			return result;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sum_M"/>
		public override float Sum_M(ISigmaDiffDataBuffer<float> value)
		{
			return Sum_V(value);
		}

		public override int MaxIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0;
			}

			float[] aData = value.Data;
			int aOffset = value.Offset, len = value.Length, maxIndex = 0;
			float maxValue = float.NegativeInfinity;

			fixed (float* aref = &aData[aOffset])
			{
				for (int i = 0; i < len; i++)
				{
					if (aref[i] > maxValue)
					{
						maxIndex = i;
						maxValue = aref[i];
					}
				}
			}

			return maxIndex;
		}

		public override int MinIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return 0;
			}

			float[] aData = value.Data;
			int aOffset = value.Offset, len = value.Length, minIndex = 0;
			float minValue = float.PositiveInfinity;

			fixed (float* aref = &aData[aOffset])
			{
				for (int i = 0; i < len; i++)
				{
					if (aref[i] < minValue)
					{
						minIndex = i;
						minValue = aref[i];
					}
				}
			}

			return minIndex;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_V_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_V_V_InPlace"/>
		public override ISigmaDiffDataBuffer<float> Add_V_V_InPlace(ISigmaDiffDataBuffer<float> a, int aOffset, ISigmaDiffDataBuffer<float> b, int bOffset, int len)
		{
			if (len == 0)
			{
				return b;
			}

			fixed (float* aref = &a.Data[a.Offset + aOffset])
			fixed (float* bref = &b.Data[b.Offset + bOffset])
			{
				int inca = 1, incb = 1;
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, aref, &inca, bref, &incb);
			}

			return b;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_S_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_V_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_S_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_V_S"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_S_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_M_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_M_V_Add_V"/>
		public override ISigmaDiffDataBuffer<float> Mul_M_V_Add_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b, ISigmaDiffDataBuffer<float> obj2)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_Dot_V_V"/>
		public override float Mul_Dot_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> n)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_V_M"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Solve_M_V"/>
		public override FSharpOption<ISigmaDiffDataBuffer<float>> Solve_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.SolveSymmetric_M_V"/>
		public override FSharpOption<ISigmaDiffDataBuffer<float>> SolveSymmetric_M_V(ShapedDataBufferView<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Diagonal_M"/>
		public override ISigmaDiffDataBuffer<float> Diagonal_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map_F_V"/>
		public override ISigmaDiffDataBuffer<float> Map_F_V(MapOp mapOp, FSharpFunc<float, float> a, ISigmaDiffDataBuffer<float> b)
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map_F_S_V"/>
		public override ISigmaDiffDataBuffer<float> Map_F_S_V(float other, MapOp mapOp, FSharpFunc<float, float> function, ISigmaDiffDataBuffer<float> value)
		{
			return Map_F_V(mapOp, function, value);
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map2_F_V_V"/>
		public override ISigmaDiffDataBuffer<float> Map2_F_V_V(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> function, ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			if (a.Length == 0)
			{
				return Map2_F_V_V(mapOp, function, CreateDataBuffer(new float[b.Length]), b);
			}
			if (b.Length == 0)
			{
				return Map2_F_V_V(mapOp, function, a, CreateDataBuffer(new float[a.Length]));
			}

			b = b.DeepCopy();

			for (int i = 0; i < a.Length; i++)
			{
				b.Data[i] = function.Invoke(a.Data[i + a.Offset]).Invoke(b.Data[i + b.Offset]);
			}

			return b;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_Out_V_V"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_M_M"/>
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

		public override ShapedDataBufferView<float> Add_M_M_InPlace(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return b;
			}
			if (b.Length == 0)
			{
				return a;
			}

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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_S_M"/>
		public override ShapedDataBufferView<float> Add_S_M(float a, ShapedDataBufferView<float> b)
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
				float alpha = 1.0f;

				BlasBackend.Saxpy(&len, &alpha, &a, &inca, bref, &incb);
			}

			return b;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Add_V_MCols"/>
		public override ShapedDataBufferView<float> Add_V_MCols(ISigmaDiffDataBuffer<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_M_M"/>
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

			a = a.DeepCopy();
			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			{
				int len = Math.Min(a.Length, b.Length);
				int inca = 1, incb = 1;
				float alpha = -1.0f;

				BlasBackend.Saxpy(&len, &alpha, bref, &incb, aref, &inca);
			}

			return a;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_M_S"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Sub_S_M"/>
		public override ShapedDataBufferView<float> Sub_S_M(float a, ShapedDataBufferView<float> b)
		{
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			b = b.DeepCopy();
			// note: doesn't work like this with blas since result inc cannot be 0
			//fixed (float* bref = &b.DataBuffer.Data[b.DataBuffer.Offset])
			//{
			//	int len = b.Length;
			//	int inca = 0, incb = 1;
			//	float alpha = -1.0f;

			//	BlasBackend.Saxpy(&len, &alpha, bref, &incb, &a, &inca);
			//}

			// TODO update with blas implementation if applicable
			//  it doesn't work like this because the result second parameter y cannot have an inc of 0 because then the result isn't actually stored
			float[] data = b.DataBuffer.Data;
			int offset = b.DataBuffer.Offset;
			int len = b.DataBuffer.Length + offset;

			for (int i = offset; i < len; i++)
			{
				data[i] = a - data[i];
			}

			return b;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_M_M"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_S_M"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_M_M_Add_V_MCols"/>
		public override ShapedDataBufferView<float> Mul_M_M_Add_V_MCols(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b, ISigmaDiffDataBuffer<float> c)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Mul_Had_M_M"/>
		public override unsafe ShapedDataBufferView<float> Mul_Had_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[b.Length]), b.Shape);
			}
			if (b.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[a.Length]), a.Shape);
			}

			int len = Math.Min(a.Length, b.Length);
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), (long[])b.Shape.Clone());

			float[] aData = a.DataBuffer.Data, bData = b.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, bOffset = b.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* bref = &bData[bOffset])
			fixed (float* resref = &resData[resOffset])
			{
				int simdLength = Vector<float>.Count, i;

				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					Vector<float> vb = new Vector<float>(bData, i + bOffset);
					(va * vb).CopyTo(resData, i + resOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = aref[i] * bref[i];
				}
			}

			return result;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Inverse_M"/>
		public override FSharpOption<ShapedDataBufferView<float>> Inverse_M(ShapedDataBufferView<float> a)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Det_M"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Transpose_M"/>
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

			fixed (float* aref = &a.DataBuffer.Data[a.DataBuffer.Offset])
			fixed (float* bref = &transposed.DataBuffer.Data[transposed.DataBuffer.Offset])
			{
				int ordering = 101; // CBLAS_LAYOUT - CblasRowMajor
				int trans = 112; // CBLAS_TRANSPOSE - CblasTrans
				int rows = a.Rows, cols = a.Cols;
				int lda = a.Cols, ldb = a.Rows;
				float alpha = 1.0f;

				BlasBackend.Somatcopy(ordering, trans, rows, cols, alpha, aref, lda, bref, ldb);
			}

			return transposed;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Permute_M"/>
		public override ShapedDataBufferView<float> Permute_M(ShapedDataBufferView<float> a, int[] permutedDimensions)
		{
			long[] originalShape = a.Shape;
			long[] permutedShape = ArrayUtils.PermuteArray(originalShape, permutedDimensions);
			long[] originalStrides = NDArrayUtils.GetStrides(originalShape), permutedStrides = NDArrayUtils.GetStrides(permutedShape);
			int rank = originalShape.Length, unitSize = 1;

			for (int i = permutedDimensions.Length - 1; i >= 0; i--)
			{
				if (permutedDimensions[i] != i)
				{
					break;
				}

				unitSize = checked (unitSize * (int) originalShape[i]);
			}

			int unitSizeBytes = unitSize * sizeof(float), len = a.Length;
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), permutedShape);
			float[] aData = a.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				long[] bufferIndices = new long[rank];

				if (unitSize > 1) // TODO maybe set a limit for when block copy outperforms manual copy (don't forget to change i++ in else branc)
				{
					for (int i = 0; i < len; i += unitSize)
					{
						NDArrayUtils.GetIndices(i, originalShape, originalStrides, bufferIndices);

						bufferIndices = ArrayUtils.PermuteArray(bufferIndices, permutedDimensions);

						int resultIndex = (int) NDArrayUtils.GetFlatIndex(permutedShape, permutedStrides, bufferIndices);

						Buffer.BlockCopy(aData, aOffset * sizeof(float), resData, (resultIndex + resOffset) * sizeof(float), unitSizeBytes);
					}
				}
				else
				{
					for (int i = 0; i < len; i++)
					{
						NDArrayUtils.GetIndices(i, originalShape, originalStrides, bufferIndices);

						bufferIndices = ArrayUtils.PermuteArray(bufferIndices, permutedDimensions);

						int resultIndex = (int) NDArrayUtils.GetFlatIndex(permutedShape, permutedStrides, bufferIndices);

						resref[resultIndex] = aref[i];
					}
				}
			}

			return result;
		}

		public override ShapedDataBufferView<float> Reshape_M(ShapedDataBufferView<float> array, long[] newShape)
		{
			ShapedDataBufferView<float> reshaped = new ShapedDataBufferView<float>(array.DataBuffer, newShape);

			return reshaped;
		}

		private bool _InternalOptimisedMapOp_F_M(MapOp mapOp, ref ShapedDataBufferView<float> a)
		{
			if (mapOp.IsExp)
			{
				_InternalOptimisedExp(ref a);

				return true;
			}
			else if (mapOp.IsSqrt)
			{
				_InternalOptimisedSqrt(ref a);

				return true;
			}
			else if (mapOp.IsSign)
			{
				_InternalOptimisedSign(ref a);

				return true;
			}
			else if (mapOp.IsReL)
			{
				_InternalOptimisedRel(ref a);

				return true;
			}
			else if (mapOp.IsSigmoid)
			{
				_InternalOptimisedSigmoid(ref a);

				return true;
			}
			else if (mapOp.IsLog)
			{
				_InternalOptimisedLog(ref a);

				return true;
			}

			return false;
		}

		private static void _InternalOptimisedSign(ref ShapedDataBufferView<float> a)
		{
			a = a.DeepCopy();
			int upper = a.DataBuffer.Offset + a.DataBuffer.Length;
			float[] data = a.DataBuffer.Data;

			System.Array.Clear(data, a.DataBuffer.Offset, a.DataBuffer.Length);

			for (int i = a.DataBuffer.Offset; i < upper; i++)
			{
				float value = data[i];

				if (value > 0.0f)
				{
					data[i] = 1.0f;
				}
				else if (value < 0.0f)
				{
					data[i] = -1.0f;
				}
			}
		}

		private void _InternalOptimisedSqrt(ref ShapedDataBufferView<float> a)
		{
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			float[] aData = a.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;
			int len = a.Length;

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				int simdLength = Vector<float>.Count, i;

				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					Vector.SquareRoot(va).CopyTo(resData, i + resOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = (float)Math.Sqrt(aref[i]);
				}
			}

			a = result; // write out result to ref
		}

		private static void _InternalOptimisedExp(ref ShapedDataBufferView<float> a)
		{
			a = a.DeepCopy();
			int upper = a.DataBuffer.Offset + a.DataBuffer.Length;
			float[] data = a.DataBuffer.Data;

			for (int i = a.DataBuffer.Offset; i < upper; i++)
			{
				data[i] = (float)Math.Exp(data[i]);
			}
		}

		private void _InternalOptimisedRel(ref ShapedDataBufferView<float> a)
		{
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			int simdLength = Vector<float>.Count;
			int len = a.Length;
			float[] aData = a.DataBuffer.Data, cData = CreateValueArray(simdLength, 0.0f), resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;
			Vector<float> vc = new Vector<float>(cData);

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				int i;
				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					Vector.Max(va, vc).CopyTo(resData, i + resOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = Math.Max(aref[i], 0.0f);
				}
			}

			a = result;
		}

		// number1 / (number1 + exp -v))
		private void _InternalOptimisedSigmoid(ref ShapedDataBufferView<float> a)
		{
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			int simdLength = Vector<float>.Count;
			int len = a.Length;
			float[] aData = a.DataBuffer.Data, cData = CreateValueArray(simdLength, 1.0f), resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;
			Vector<float> vc = new Vector<float>(cData);

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				for (int y = 0; y < len; y++)
				{
					resref[y] = (float)Math.Exp(-aref[y]);
				}

				int i;
				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					int absResOffset = i + resOffset;
					Vector<float> vres = new Vector<float>(resData, absResOffset); // TODO optimise offsets
					vres = vres + vc;
					(vc / vres).CopyTo(resData, absResOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = 1.0f / (1.0f + resref[i]);
				}
			}

			a = result;
		}

		private void _InternalOptimisedLog(ref ShapedDataBufferView<float> a)
		{
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(a.Length)), (long[])a.Shape.Clone());

			int len = a.Length;
			float[] aData = a.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				for (int i = 0; i < len; i++)
				{
					resref[i] = (float) Math.Log(aref[i]);
				}
			}

			a = result;
		}

		private bool _InternalOptimisedMapOp_F_S_M(float other, MapOp mapOp, ref ShapedDataBufferView<float> a)
		{
			if (mapOp.IsDiv)
			{
				_InternalOptimisedDiv(other, ref a);

				return true;
			}

			return false;
		}

		private void _InternalOptimisedDiv(float other, ref ShapedDataBufferView<float> a)
		{
			int simdLength = Vector<float>.Count;
			int len = a.Length;
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), (long[])a.Shape.Clone());
			float[] aData = a.DataBuffer.Data, bData = CreateValueArray(simdLength, other), resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;
			Vector<float> vc = new Vector<float>(bData); // filled with constant value of other

			fixed (float* aref = &aData[aOffset])
			fixed (float* resref = &resData[resOffset])
			{
				int i;
				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					(vc / va).CopyTo(resData, i + resOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = other / aref[i];
				}
			}

			a = result;
		}

		private bool _InternalOptimisedMapOp_F_M_M(MapOp mapOp, ShapedDataBufferView<float> a, ref ShapedDataBufferView<float> b)
		{
			if (mapOp.IsDiv)
			{
				_InternalOptimisedDiv(a, ref b);

				return true;
			}

			return false;
		}

		private void _InternalOptimisedDiv(ShapedDataBufferView<float> a, ref ShapedDataBufferView<float> b)
		{
			int len = Math.Min(a.Length, b.Length);
			ShapedDataBufferView<float> result = new ShapedDataBufferView<float>(CreateDataBuffer(CreateUninitialisedArray(len)), (long[])b.Shape.Clone());

			float[] aData = a.DataBuffer.Data, bData = b.DataBuffer.Data, resData = result.DataBuffer.Data;
			int aOffset = a.DataBuffer.Offset, bOffset = b.DataBuffer.Offset, resOffset = result.DataBuffer.Offset;

			fixed (float* aref = &aData[aOffset])
			fixed (float* bref = &bData[bOffset])
			fixed (float* resref = &resData[resOffset])
			{
				int simdLength = Vector<float>.Count, i;

				for (i = 0; i <= len - simdLength; i += simdLength)
				{
					Vector<float> va = new Vector<float>(aData, i + aOffset); // TODO optimise offsets
					Vector<float> vb = new Vector<float>(bData, i + bOffset);
					(va / vb).CopyTo(resData, i + resOffset);
				}

				for (; i < len; ++i)
				{
					resref[i] = aref[i] / bref[i];
				}
			}

			b = result;
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map_F_M"/>
		public override ShapedDataBufferView<float> Map_F_M(MapOp mapOp, FSharpFunc<float, float> f, ShapedDataBufferView<float> a)
		{
			if (a.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			if (_InternalOptimisedMapOp_F_M(mapOp, ref a))
			{
				return a;
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map_F_S_M"/>
		public override ShapedDataBufferView<float> Map_F_S_M(float other, MapOp mapOp, FSharpFunc<float, float> function, ShapedDataBufferView<float> value)
		{
			if (_InternalOptimisedMapOp_F_S_M(other, mapOp, ref value))
			{
				return value;
			}

			return Map_F_M(mapOp, function, value); // this is correct, the "other" constant is only passed to speed up computation in optimised versions without using the lambda
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.Map2_F_M_M"/>
		public override ShapedDataBufferView<float> Map2_F_M_M(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> f, ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.ReshapeCopy_MRows_V"/>
		public override ISigmaDiffDataBuffer<float> ReshapeCopy_MRows_V(ShapedDataBufferView<float> value)
		{
			if (value.Length == 0)
			{
				return CreateDataBuffer(new float[0]);
			}

			return value.DataBuffer.DeepCopy();
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.ReshapeCopy_V_MRows"/>
		public override ShapedDataBufferView<float> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> value)
		{
			if (value.Length == 0)
			{
				return new ShapedDataBufferView<float>(CreateDataBuffer(new float[0]), 0L, 0L);
			}

			int n = value.Length / rows;

			return new ShapedDataBufferView<float>(value.DeepCopy(), rows, n);
		}

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.RepeatReshapeCopy_V_MRows"/>
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

		/// <inheritdoc cref="DiffSharpBackendHandle{T}.RepeatReshapeCopy_V_MCols"/>
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}
	}
}
