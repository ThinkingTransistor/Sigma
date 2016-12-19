/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp.Backend;
using static DiffSharp.Util;
using Microsoft.FSharp.Core;

namespace Sigma.Core.Handlers.Backends.SigmaDiff
{
	public abstract class DiffSharpBackendHandle<T> : Backend<T>
	{
		public IBlasBackend BlasBackend { get; set; }
		public ILapackBackend LapackBackend { get; set; }

		internal DiffSharpBackendHandle(IBlasBackend blasBackend, ILapackBackend lapackBackend)
		{
			if (blasBackend == null) throw new ArgumentNullException(nameof(blasBackend));
			if (lapackBackend == null) throw new ArgumentNullException(nameof(lapackBackend));

			BlasBackend = blasBackend;
			LapackBackend = lapackBackend;
		}

		public abstract ISigmaDiffDataBuffer<T> CreateDataBuffer(T[] values);
		public abstract T Mul_Dot_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> n);
		public abstract T L1Norm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T L2Norm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T SupNorm_V(ISigmaDiffDataBuffer<T> value);
		public abstract T Sum_V(ISigmaDiffDataBuffer<T> value);
		public abstract T Sum_M(ISigmaDiffDataBuffer<T> value);
		public abstract ISigmaDiffDataBuffer<T> Add_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Add_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Sub_V_S(ISigmaDiffDataBuffer<T> a, T b);
		public abstract ISigmaDiffDataBuffer<T> Mul_S_V(T a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Mul_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Mul_M_V_Add_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b, ISigmaDiffDataBuffer<T> obj2);
		public abstract ISigmaDiffDataBuffer<T> Mul_V_M(ISigmaDiffDataBuffer<T> a, ShapedDataBufferView<T> b);
		public abstract FSharpOption<ISigmaDiffDataBuffer<T>> Solve_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract FSharpOption<ISigmaDiffDataBuffer<T>> SolveSymmetric_M_V(ShapedDataBufferView<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Diagonal_M(ShapedDataBufferView<T> a);
		public abstract ISigmaDiffDataBuffer<T> Map_F_V(FSharpFunc<T, T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ISigmaDiffDataBuffer<T> Map2_F_V_V(FSharpFunc<T, FSharpFunc<T, T>> a, ISigmaDiffDataBuffer<T> b, ISigmaDiffDataBuffer<T> obj2);
		public abstract ISigmaDiffDataBuffer<T> ReshapeCopy_MRows_V(ShapedDataBufferView<T> value);
		public abstract ShapedDataBufferView<T> Mul_Out_V_V(ISigmaDiffDataBuffer<T> a, ISigmaDiffDataBuffer<T> b);
		public abstract ShapedDataBufferView<T> Add_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Add_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Add_V_MCols(ISigmaDiffDataBuffer<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Sub_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Sub_M_S(ShapedDataBufferView<T> a, T b);
		public abstract ShapedDataBufferView<T> Sub_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_S_M(T a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Mul_M_M_Add_V_MCols(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b, ISigmaDiffDataBuffer<T> obj2);
		public abstract ShapedDataBufferView<T> Mul_Had_M_M(ShapedDataBufferView<T> a, ShapedDataBufferView<T> b);
		public abstract FSharpOption<ShapedDataBufferView<T>> Inverse_M(ShapedDataBufferView<T> a);
		public abstract FSharpOption<T> Det_M(ShapedDataBufferView<T> a);
		public abstract ShapedDataBufferView<T> Transpose_M(ShapedDataBufferView<T> a);
		public abstract ShapedDataBufferView<T> Map_F_M(FSharpFunc<T, T> a, ShapedDataBufferView<T> b);
		public abstract ShapedDataBufferView<T> Map2_F_M_M(FSharpFunc<T, FSharpFunc<T, T>> a, ShapedDataBufferView<T> b, ShapedDataBufferView<T> obj2);
		public abstract ShapedDataBufferView<T> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<T> value);
		public abstract ShapedDataBufferView<T> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<T> value);
		public abstract ShapedDataBufferView<T> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<T> value);
	}
}
