/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;
using Sigma.Core.Data;
using Sigma.Core.Math;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.Serialization;

namespace Sigma.Core.Handlers.Backends
{
	/// <summary>
	/// A computation handler that runs computations on the CPU with 32-bit floating point precision. 
	/// </summary>
	public class CPUFloat32Handler : IComputationHandler
	{
		public IDataType DataType { get { return DataTypes.FLOAT32; } }

		public CPUFloat32Handler()
		{
		}

		public INDArray Create(params long[] shape)
		{
			return new NDArray<float>(shape: shape);
		}

		public void InitAfterDeserialisation(INDArray array)
		{
			// nothing to do here for this handler, all relevant components are serialised automatically
		}

		public long GetSizeBytes(params INDArray[] arrays)
		{
			long totalSizeBytes = 0L;

			foreach (INDArray array in arrays)
			{
				totalSizeBytes += System.Runtime.InteropServices.Marshal.SizeOf(array);
			}

			return totalSizeBytes;
		}

		public bool IsInterchangeable(IComputationHandler otherHandler)
		{
			//there are no interchangeable implementations so it will have to be the same type 
			return otherHandler.GetType() == this.GetType();
		}

		public bool CanConvert(INDArray array, IComputationHandler otherHandler)
		{
			//if it's the same base unit and at least the same precision we can convert
			return otherHandler.DataType.BaseUnderlyingType == this.DataType.BaseUnderlyingType && otherHandler.DataType.SizeBytes >= this.DataType.SizeBytes;
		}

		public INDArray Convert(INDArray array, IComputationHandler otherHandler)
		{
			return new NDArray<float>(array.GetDataAs<float>(), array.Shape);
		}

		public void Fill(INDArray arrayToFill, INDArray filler)
		{
			throw new NotImplementedException();
		}
	}
}
