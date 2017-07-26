/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using DiffSharp.Interop.Float32;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract;
using System;
using Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu;

namespace Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu
{
	/// <summary>
	/// A computation handler that runs computations on the CPU with 32-bit floating point precision. 
	/// </summary>
	[Serializable]
	public class CpuFloat32Handler : DiffSharpFloat32Handler<ADFloat32NDArray, ADFloat32Number>
	{
		/// <inheritdoc />
		public CpuFloat32Handler() : base(new OpenBlasBlasBackend(), new OpenBlasLapackBackend())
		{
		}

		/// <inheritdoc />
		public override IDataType DataType => DataTypes.Float32;

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public override void OnDeserialised()
		{
			InitialiseBackend(new DiffSharpFloat32BackendHandle(BlasBackend, LapackBackend, backendTag: -1));
		}

		/// <inheritdoc />
		public override IDataBuffer<T> DataBuffer<T>(T[] values)
		{
			return new SigmaDiffDataBuffer<T>(values, backendTag: DiffsharpBackendHandle.BackendTag);
		}

		/// <inheritdoc />
		public override INDArray NDArray(params long[] shape)
		{
			return AssignTag(new ADFloat32NDArray(DiffsharpBackendHandle.BackendTag, shape)).SetAssociatedHandler(this);
		}

		/// <inheritdoc />
		public override INDArray NDArray<TOther>(TOther[] values, params long[] shape)
		{
			float[] convertedValues = new float[values.Length];
			Type floatType = typeof(float);

			for (int i = 0; i < values.Length; i++)
			{
				convertedValues[i] = (float)System.Convert.ChangeType(values[i], floatType);
			}

			return AssignTag(new ADFloat32NDArray(DiffsharpBackendHandle.BackendTag, convertedValues, shape)).SetAssociatedHandler(this);
		}

		/// <inheritdoc />
		public override INumber Number(object value)
		{
			return new ADFloat32Number((float)System.Convert.ChangeType(value, typeof(float))).SetAssociatedHandler(this);
		}

		/// <inheritdoc />
		public override INDArray AsNDArray(INumber number)
		{
			ADFloat32Number internalNumber = InternaliseNumber(number);

			return AssignTag(new ADFloat32NDArray(DNDArray.OfDNumber(internalNumber.Handle, DiffsharpBackendHandle)));
		}

		/// <inheritdoc />
		public override INumber AsNumber(INDArray array, params long[] indices)
		{
			ADFloat32NDArray internalArray = InternaliseArray(array);
			long flatIndex = NDArrayUtils.GetFlatIndex(array.Shape, array.Strides, indices);

			return new ADFloat32Number(DNDArray.ToDNumber(internalArray.Handle, (int)flatIndex));
		}

		/// <inheritdoc />
		public override void InitAfterDeserialisation(INDArray array)
		{
			// nothing to do here for this handler, all relevant components are serialised automatically, 
			// diffsharp does not need to be de-serialised, components only need to be removed from trace
		}

		/// <inheritdoc />
		public override long GetSizeBytes(params INDArray[] arrays)
		{
			long totalSizeBytes = 0L;

			foreach (INDArray array in arrays)
			{
				long sizeBytes = 32L; // let's just assume this many bytes of base fluff, I really have no idea

				sizeBytes += array.Length * DataType.SizeBytes;
				sizeBytes += (array.Shape.Length) * 8L * 2;

				totalSizeBytes += sizeBytes;
			}

			return totalSizeBytes;
		}

		/// <inheritdoc />
		public override bool IsOwnFormat(INDArray array)
		{
			return array is ADFloat32NDArray;
		}

		/// <inheritdoc />
		public override bool IsInterchangeable(IComputationHandler otherHandler)
		{
			//there are no interchangeable implementations so it will have to be the same type 
			return otherHandler.GetType() == GetType();
		}

		/// <inheritdoc />
		public override bool CanConvert(INDArray array, IComputationHandler otherHandler)
		{
			//if it's the same base unit and at least the same precision we can convert
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
			IDataBuffer<float> arrayToFillData = InternaliseArray(arrayToFill).Data;
			IDataBuffer<float> fillerData = InternaliseArray(filler).Data;

			arrayToFillData.SetValues(fillerData.Data, fillerData.Offset, 0, Math.Min(arrayToFill.Length, filler.Length));
		}

		/// <inheritdoc />
		public override void Fill<TOther>(TOther value, INDArray arrayToFill)
		{
			IDataBuffer<float> arrayToFillData = InternaliseArray(arrayToFill).Data;

			float floatValue = (float)System.Convert.ChangeType(value, typeof(float));

			for (int i = 0; i < arrayToFillData.Length; i++)
			{
				arrayToFillData.Data.SetValue(floatValue, i);
			}
		}

		/// <inheritdoc />
		public override void Fill(INDArray filler, INDArray arrayToFill, long[] sourceBeginIndices, long[] sourceEndIndices, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			IDataBuffer<float> fillerData = InternaliseArray(filler).Data;
			IDataBuffer<float> arrayToFillData = InternaliseArray(arrayToFill).Data;

			int sourceOffset = (int)NDArrayUtils.GetFlatIndex(filler.Shape, filler.Strides, sourceBeginIndices);
			int sourceLength = (int)NDArrayUtils.GetFlatIndex(filler.Shape, filler.Strides, sourceEndIndices) - sourceOffset + 1; // +1 because end is inclusive
			int destinationOffset = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationBeginIndices);
			int destinationLength = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationEndIndices) - destinationOffset + 1; // same here

			if (sourceLength < 0) throw new ArgumentOutOfRangeException($"Source begin indices must be smaller than source end indices, but source length was {sourceLength}.");
			if (destinationLength < 0) throw new ArgumentOutOfRangeException($"Destination begin indices must be smaller than destination end indices, but destination length was {destinationLength}.");
			if (sourceLength != destinationLength) throw new ArgumentException($"Source and destination indices length must batch, but source length was {sourceLength} and destination legnth was {destinationLength}.");

			Array.Copy(fillerData.Data, sourceOffset, arrayToFillData.Data, destinationOffset, sourceLength);
		}

		/// <inheritdoc />
		public override void Fill<T>(T[] filler, INDArray arrayToFill, long[] destinationBeginIndices, long[] destinationEndIndices)
		{
			IDataBuffer<float> arrayToFillData = InternaliseArray(arrayToFill).Data;

			int destinationOffset = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationBeginIndices);
			int destinationLength = (int)NDArrayUtils.GetFlatIndex(arrayToFill.Shape, arrayToFill.Strides, destinationEndIndices) - destinationOffset + 1; // +1 because end is inclusive

			if (destinationLength < 0) throw new ArgumentOutOfRangeException($"Destination begin indices must be smaller than destination end indices, but destination length was {destinationLength}.");

			Array.Copy(filler, 0, arrayToFillData.Data, destinationOffset, destinationLength);
		}

		/// <inheritdoc />
		protected override ADFloat32NDArray CreateArrayFromHandle(DNDArray handle)
		{
			return new ADFloat32NDArray(handle);
		}

		/// <inheritdoc />
		protected override ADFloat32Number CreateNumberFromHandle(DNumber handle)
		{
			return new ADFloat32Number(handle);
		}

		/// <inheritdoc />
		protected override ADFloat32NDArray ConvertInternal(INDArray array)
		{
			ADFloat32NDArray @internal = array as ADFloat32NDArray;

			if (@internal != null) return @internal;

			return new ADFloat32NDArray(DiffsharpBackendHandle.BackendTag, array.GetDataAs<float>(), array.Shape);
		}
	}
}
