/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Diagnostics.CodeAnalysis;
using DiffSharp.Interop.Float32;
using Sigma.Core.Handlers.Backends.SigmaDiff;

namespace Sigma.Core.MathAbstract.Backends.SigmaDiff.NativeCpu
{
	/// <summary>
	/// A number with a float32 CPU-based in-system-memory backend Sigma.DiffSharp handle for tracing and AD operations.
	/// </summary>
	[SuppressMessage("ReSharper", "InconsistentNaming")]
	[Serializable]
	public class ADFloat32Number : ADNumber<float>, IADFloat32NumberHandle
	{
		/// <inheritdoc />
		public DNumber Handle { get; private set; }

		public ADFloat32Number(float value) : base(value)
		{
			Handle = new DNumber(value);
		}

		public ADFloat32Number(DNumber numberHandle) : base(numberHandle.Value)
		{
			Handle = numberHandle;
		}

		internal override void SetValue(float value)
		{
			base.SetValue(value);

			Handle = new DNumber(value);
		}
	}
}
