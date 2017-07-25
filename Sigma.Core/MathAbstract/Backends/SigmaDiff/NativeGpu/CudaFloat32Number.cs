/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeGpu
{
	/// <summary>
	/// A number with a float32 CUDA-based in-GPU-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	public class CudaFloat32Number : ADNumber<float>
	{
		/// <summary>
		/// Create a single value (i.e. number) with a certain initial value.
		/// </summary>
		/// <param name="value">The initial value to wrap.</param>
		public CudaFloat32Number(float value) : base(value)
		{
		}
	}
}
