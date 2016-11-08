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
		private IFormatter serialisationFormatter;

		public CPUFloat32Handler()
		{
			this.serialisationFormatter = new BinaryFormatter();
		}

		public INDArray Create(params long[] shape)
		{
			return new NDArray<float>(shape: shape);
		}

		public INDArray Deserialise(Stream stream)
		{
			return (INDArray) serialisationFormatter.Deserialize(stream);
		}

		public void Serialise(INDArray array, Stream stream)
		{
			serialisationFormatter.Serialize(stream, array);
		}

		public long GetSizeBytes(INDArray array)
		{
			return System.Runtime.InteropServices.Marshal.SizeOf(array);
		}
	}
}
