/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp.Backend;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using Microsoft.FSharp.Core;
using static DiffSharp.Util;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaFloat32BackendHandle : DiffSharpBackendHandle<float>
	{
		private readonly CudaBlas _cudaBlasHandle;
		private CudaContext _cudaContext;

		public CudaFloat32BackendHandle(int deviceId, long backendTag) : base(backendTag)
		{ 
			_cudaBlasHandle = new CudaBlas();
			_cudaContext = new CudaContext(deviceId); // TODO move control to create / destroy context methods for GpuWorkers (context is thread-bound) - begin session perhaps? 
			_cudaContext.SetCurrent();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> CreateDataBuffer(float[] values)
		{
			return new CudaSigmaDiffDataBuffer<float>(values, BackendTag, _cudaContext);
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
		public override float Sum_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override float Sum_M(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override int MaxIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override int MinIndex_V(ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> Add_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
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
		public override ShapedDataBufferView<float> Map_F_M(MapOp mapOp, FSharpFunc<float, float> function, ShapedDataBufferView<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Map_F_S_M(float other, MapOp mapOp, FSharpFunc<float, float> function, ShapedDataBufferView<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Map2_F_M_M(MapOp mapOp, FSharpFunc<float, FSharpFunc<float, float>> function, ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ISigmaDiffDataBuffer<float> ReshapeCopy_MRows_V(ShapedDataBufferView<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_Out_V_V(ISigmaDiffDataBuffer<float> a, ISigmaDiffDataBuffer<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Add_M_M_InPlace(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
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
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Sub_M_S(ShapedDataBufferView<float> a, float b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Sub_S_M(float a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_S_M(float a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_M_M_Add_V_MCols(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b, ISigmaDiffDataBuffer<float> obj2)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Mul_Had_M_M(ShapedDataBufferView<float> a, ShapedDataBufferView<float> b)
		{
			throw new NotImplementedException();
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
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Permute_M(ShapedDataBufferView<float> array, int[] rearrangedDimensions)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> Reshape_M(ShapedDataBufferView<float> array, long[] newShape)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> ReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MRows(int rows, ISigmaDiffDataBuffer<float> row)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override ShapedDataBufferView<float> RepeatReshapeCopy_V_MCols(int cols, ISigmaDiffDataBuffer<float> value)
		{
			throw new NotImplementedException();
		}
	}
}
