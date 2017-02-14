/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using DiffSharp.Interop.Float32;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.DiffSharp;
using Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu;
using System;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu
{
	/// <summary>
	/// A computation handler that runs computations on the CPU with 32-bit floating point precision. 
	/// </summary>
	public class CpuFloat32Handler : DiffSharpFloat32Handler
	{
		public CpuFloat32Handler() : base(new OpenBlasBlasBackend(), new OpenBlasLapackBackend())
		{
		}

		public override IDataType DataType => DataTypes.Float32;

		public override IDataBuffer<T> DataBuffer<T>(T[] values)
		{
			return new SigmaDiffDataBuffer<T>(values, backendTag: DiffsharpBackendHandle.BackendTag);
		}

		public override INDArray NDArray(params long[] shape)
		{
			return AssignTag(new ADNDFloat32Array(DiffsharpBackendHandle.BackendTag, shape)).SetAssociatedHandler(this);
		}

		public override INDArray NDArray<TOther>(TOther[] values, params long[] shape)
		{
			float[] convertedValues = new float[values.Length];
			Type floatType = typeof(float);

			for (int i = 0; i < values.Length; i++)
			{
				convertedValues[i] = (float) System.Convert.ChangeType(values[i], floatType);
			}

			return AssignTag(new ADNDFloat32Array(DiffsharpBackendHandle.BackendTag, convertedValues, shape)).SetAssociatedHandler(this);
		}

		public override INumber Number(object value)
		{
			return new ADFloat32Number((float) System.Convert.ChangeType(value, typeof(float))).SetAssociatedHandler(this);
		}

		public override INDArray AsNDArray(INumber number)
		{
			ADFloat32Number internalNumber = InternaliseNumber(number);

			return AssignTag(new ADNDFloat32Array(DNDArray.OfDNumber(internalNumber._adNumberHandle, DiffsharpBackendHandle)));
		}

		public override INumber AsNumber(INDArray array, params long[] indices)
		{
			ADNDFloat32Array internalArray = InternaliseArray(array);
			long flatIndex = NDArrayUtils.GetFlatIndex(array.Shape, array.Strides, indices);

			return new ADFloat32Number(DNDArray.ToDNumber(internalArray._adArrayHandle, (int) flatIndex));
		}

		public override void InitAfterDeserialisation(INDArray array)
		{
			// nothing to do here for this handler, all relevant components are serialised automatically, 
			// diffsharp does not need to be de-serialised, components only need to be removed from trace
		}

		public override long GetSizeBytes(params INDArray[] arrays)
		{
			long totalSizeBytes = 0L;

			foreach (INDArray array in arrays)
			{
				long sizeBytes = 52L; // let's just assume 52bytes of base fluff, I really have no idea

				sizeBytes += array.Length * DataType.SizeBytes;
				sizeBytes += (array.Shape.Length) * 8L * 2;

				totalSizeBytes += sizeBytes;
			}

			return totalSizeBytes;
		}

		public override bool IsInterchangeable(IComputationHandler otherHandler)
		{
			//there are no interchangeable implementations so it will have to be the same type 
			return otherHandler.GetType() == GetType();
		}

		public override bool CanConvert(INDArray array, IComputationHandler otherHandler)
		{
			//if it's the same base unit and at least the same precision we can convert
			return otherHandler.DataType.BaseUnderlyingType == DataType.BaseUnderlyingType && otherHandler.DataType.SizeBytes >= DataType.SizeBytes;
		}

		public override INDArray Convert(INDArray array, IComputationHandler otherHandler)
		{
			return ConvertInternal(array);
		}

		public override void Fill(INDArray filler, INDArray arrayToFill)
		{
			IDataBuffer<float> arrayToFillData = ((ADNDArray<float>) arrayToFill).Data;
			IDataBuffer<float> fillerData = ((ADNDArray<float>) filler).Data;

			arrayToFillData.SetValues(fillerData.Data, fillerData.Offset, arrayToFillData.Offset, Math.Min(arrayToFill.Length, filler.Length));
		}

		public override void Fill<TOther>(TOther value, INDArray arrayToFill)
		{
			IDataBuffer<float> arrayToFillData = ((ADNDArray<float>) arrayToFill).Data;

			float floatValue = (float) System.Convert.ChangeType(value, typeof(float));

			for (int i = 0; i < arrayToFillData.Length; i++)
			{
				arrayToFillData.Data.SetValue(floatValue, i);
			}
		}
	}
}
