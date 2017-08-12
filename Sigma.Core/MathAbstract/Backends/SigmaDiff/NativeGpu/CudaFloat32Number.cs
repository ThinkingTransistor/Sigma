/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using DiffSharp.Interop.Float32;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using Sigma.Core.Handlers.Backends.SigmaDiff;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeGpu;
using Sigma.Core.Persistence;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeGpu
{
	/// <summary>
	/// A number with a float32 CUDA-based in-GPU-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	public class CudaFloat32Number : ADNumber<float>, IADFloat32NumberHandle, ISerialisationNotifier
	{
		/// <inheritdoc />
		public DNumber Handle { get; private set; }

		[NonSerialized]
		internal CudaContext CudaContext;

		private int _cudaContextDeviceId;

		public CudaFloat32Number(DNumber numberHandle, CudaContext context) : base(numberHandle.Value)
		{
			if (numberHandle == null) throw new ArgumentNullException(nameof(numberHandle));
			if (context == null) throw new ArgumentNullException(nameof(context));

			Handle = numberHandle;
			CudaContext = context;

			_cudaContextDeviceId = CudaContext.DeviceId;
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			CudaContext restoredContext = CudaFloat32Handler.GetContextForDeviceId(_cudaContextDeviceId);

			_cudaContextDeviceId = restoredContext.DeviceId;

			CudaContext = restoredContext;
		}

		internal override void SetValue(float value)
		{
			base.SetValue(value);

			Handle = new DNumber(value);
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}
	}
}
