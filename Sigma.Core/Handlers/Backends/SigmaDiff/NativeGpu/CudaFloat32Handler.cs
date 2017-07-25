/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu
{
	public class CudaFloat32Handler : DiffSharpFloat32Handler
	{
		public CudaFloat32Handler(IBlasBackend blasBackend, ILapackBackend lapackBackend) : base(blasBackend, lapackBackend)
		{
		}

		/// <summary>The underlying data type processed and used in this computation handler.</summary>
		public override IDataType DataType { get; }

		/// <inheritdoc />
		public override void InitAfterDeserialisation(INDArray array)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override long GetSizeBytes(params INDArray[] array)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override bool IsInterchangeable(IComputationHandler otherHandler)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INDArray NDArray(params long[] shape)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INDArray NDArray<TOther>(TOther[] values, params long[] shape)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INumber Number(object value)
		{
			throw new NotImplementedException();
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
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override INDArray Convert(INDArray array, IComputationHandler otherHandler)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override void Fill(INDArray filler, INDArray arrayToFill)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override void Fill(INDArray filler, INDArray arrayToFill, long[] sourceBeginIndices, long[] sourceEndIndices, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override void Fill<T>(T[] filler, INDArray arrayToFill, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			throw new NotImplementedException();
		}

		/// <inheritdoc />
		public override void Fill<TOther>(TOther value, INDArray arrayToFill)
		{
			throw new NotImplementedException();
		}
	}
}
