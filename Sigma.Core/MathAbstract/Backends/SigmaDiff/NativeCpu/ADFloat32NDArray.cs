/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using DiffSharp.Interop.Float32;
using Sigma.Core.Data;
using Sigma.Core.Handlers.Backends.SigmaDiff;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu
{
	/// <summary>
	/// An ndarray with a float32 CPU-based in-system-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	[Serializable]
	public class ADFloat32NDArray : ADNDArray<float>, IADFloat32NDArrayHandle
	{
		public DNDArray Handle { get; private set; }

		public ADFloat32NDArray(long backendTag, params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(ArrayUtils.Product(shape), backendTag), NDArrayUtils.CheckShape(shape)))
		{
		}

		public ADFloat32NDArray(long backendTag, float[] data, params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(data, backendTag), NDArrayUtils.CheckShape(shape)))
		{
		}

		public ADFloat32NDArray(DNDArray adArrayHandle) : base((IDataBuffer<float>)adArrayHandle.Buffer.DataBuffer, adArrayHandle.Buffer.Shape)
		{
			if (adArrayHandle == null) throw new ArgumentNullException(nameof(adArrayHandle));

			Handle = adArrayHandle;
			Handle.Buffer.Shape = Shape;
		}

		public ADFloat32NDArray(long backendTag, IDataBuffer<float> buffer, long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(buffer, 0, buffer.Length, backendTag), NDArrayUtils.CheckShape(shape)))
		{
		}

		protected override void Reinitialise(long[] shape, long[] strides)
		{
			Handle = Handle.ShallowCopy();

			base.Reinitialise(shape, strides);

			Handle.Buffer.Shape = shape;
		}

		/// <summary>
		/// Get a slice of this ndarray of a certain region as a new ndarray with the same underlying data.
		/// </summary>
		/// <param name="beginIndices">The begin indices (inclusively, where the slice should begin).</param>
		/// <param name="endIndices">The end indices (exclusively, where the slice should end).</param>
		/// <returns></returns>
		public override INDArray Slice(long[] beginIndices, long[] endIndices)
		{
			long[] slicedShape = GetSlicedShape(beginIndices, endIndices);

			//we want the end indices to be inclusive for easier handling
			endIndices = endIndices.Select(i => i - 1).ToArray();

			long absoluteBeginOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, beginIndices);
			long absoluteEndOffset = NDArrayUtils.GetFlatIndex(Shape, Strides, endIndices);
			long length = absoluteEndOffset - absoluteBeginOffset + 1;

			return new ADFloat32NDArray(new DNDArray(new SigmaDiffDataBuffer<float>(Data, absoluteBeginOffset, length, backendTag: ((SigmaDiffDataBuffer<float>)Data).BackendTag), slicedShape));
		}

		public override INDArray Reshape(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			return new ADFloat32NDArray(DNDArray.Reshape(Handle, newShape));
		}

		public override object DeepCopy()
		{
			return new ADFloat32NDArray(Handle.DeepCopy());
		}
	}
}
