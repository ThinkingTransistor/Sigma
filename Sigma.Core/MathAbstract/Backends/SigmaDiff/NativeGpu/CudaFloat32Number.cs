/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using DiffSharp.Interop.Float32;
using ManagedCuda;
using Sigma.Core.Handlers.Backends.SigmaDiff;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeGpu
{
	/// <summary>
	/// A number with a float32 CUDA-based in-GPU-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	public class CudaFloat32Number : ADNumber<float>, IADFloat32NumberHandle
	{
		/// <inheritdoc />
		public DNumber Handle { get; private set; }

		internal readonly CudaDeviceVariable<float> CudaBuffer;

		public CudaFloat32Number(float value) : base(value)
		{
			Handle = new DNumber(value);
			CudaBuffer = new[] { value };
		}

		public CudaFloat32Number(DNumber numberHandle) : base(numberHandle.Value)
		{
			Handle = numberHandle;
			CudaBuffer = new[] { numberHandle.Value };
		}

		internal override void SetValue(float value)
		{
			base.SetValue(value);

			Handle = new DNumber(value);
			CudaBuffer[0] = value;
		}
	}
}
