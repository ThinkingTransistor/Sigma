/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Math;
using Sigma.Core.Data;

namespace Sigma.Core.Handlers.Backends
{
	/// <summary>
	/// A computation handler that runs computations on the CPU with 32-bit floating point precision. 
	/// </summary>
	public class CPUFloat32Handler : IComputationHandler
	{
		public IDataType DataType { get { return DataTypes.FLOAT32; } }

		public INDArray Create(params long[] shape)
		{
			return new NDArray<float>(shape: shape);
		}
	}
}
