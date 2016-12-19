/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

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
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	[Serializable]
	public class ADNDFloat32Array : ADNDArray<float>
	{
		public readonly DNDArray _adArrayHandle;

		public ADNDFloat32Array(params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(ArrayUtils.Product(shape)), shape))
		{
		}

		public ADNDFloat32Array(float[] data, params long[] shape) : this(new DNDArray(new SigmaDiffDataBuffer<float>(data), shape))
		{
		}

		public ADNDFloat32Array(DNDArray adArrayHandle) : base((IDataBuffer<float>) adArrayHandle.Buffer.DataBuffer, adArrayHandle.Buffer.Shape)
		{
			if (adArrayHandle == null) throw new ArgumentNullException(nameof(adArrayHandle));

			_adArrayHandle = adArrayHandle;
		}
	}
}
