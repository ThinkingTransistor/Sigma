/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics.CodeAnalysis;
using DiffSharp.Interop.Float32;
using Sigma.Core.Data;
using Sigma.Core.Handlers.Backends.SigmaDiff;
using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu
{
	/// <summary>
	/// An ndarray with a float32 backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	[Serializable]
	public class ADNDFloat32Array : ADNDArray<float>
	{
		public DNDArray _adArrayHandle;

		public ADNDFloat32Array(long backendTag, params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(ArrayUtils.Product(shape), backendTag), shape))
		{
		}

		public ADNDFloat32Array(long backendTag, float[] data, params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(data, backendTag), shape))
		{
		}

		public ADNDFloat32Array(DNDArray adArrayHandle) : base((IDataBuffer<float>) adArrayHandle.Buffer.DataBuffer, adArrayHandle.Buffer.Shape)
		{
			if (adArrayHandle == null) throw new ArgumentNullException(nameof(adArrayHandle));

			_adArrayHandle = adArrayHandle;
		}

		protected override void Reinitialise(long[] shape, long[] strides)
		{
			_adArrayHandle = _adArrayHandle.ShallowCopy();

			base.Reinitialise(shape, strides);

			Array.Copy(shape, _adArrayHandle.Buffer.Shape, shape.Length);
		}

		public override INDArray Reshape(params long[] newShape)
		{
			if (Length != ArrayUtils.Product(newShape))
			{
				throw new ArgumentException("Reshaping cannot change total ndarray length, only array shape.");
			}

			DNDArray adArrayHandleCopy = _adArrayHandle.ShallowCopy();
			Array.Copy(newShape, adArrayHandleCopy.Buffer.Shape, newShape.Length);

			return new ADNDFloat32Array(adArrayHandleCopy);
		}

		public override object DeepCopy()
		{
			return new ADNDFloat32Array(_adArrayHandle.DeepCopy());
		}
	}
}
