/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics.CodeAnalysis;
using DiffSharp.Interop.Float32;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu
{
	/// <summary>
	/// A number with a float32 CPU-based in-system-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	[Serializable]
	public class ADFloat32Number : ADNumber<float>
	{
		public DNumber _adNumberHandle;

		public ADFloat32Number(float value) : base(value)
		{
			_adNumberHandle = new DNumber(value);
		}

		public ADFloat32Number(DNumber numberHandle) : base(numberHandle.Value)
		{
			_adNumberHandle = numberHandle;
		}

		internal override void SetValue(float value)
		{
			base.SetValue(value);

			_adNumberHandle = new DNumber(value);
		}
	}
}
