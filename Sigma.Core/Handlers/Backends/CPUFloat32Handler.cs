/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data;
using Sigma.Core.Math;
using System;
using System.ComponentModel;

namespace Sigma.Core.Handlers.Backends
{
	/// <summary>
	/// A computation handler that runs computations on the CPU with 32-bit floating point precision. 
	/// </summary>
	public class CPUFloat32Handler : IComputationHandler
	{
		public IDataType DataType { get { return DataTypes.FLOAT32; } }

		private TypeConverter converter = TypeDescriptor.GetConverter(typeof(float));

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
				long sizeBytes = 52L; // let's just assume 52bytes of base fluff, I really have no idea

				sizeBytes += array.Length * this.DataType.SizeBytes;
				sizeBytes += (array.Shape.Length) * 8L * 2;

				totalSizeBytes += sizeBytes;
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

		public void Add<TOther>(INDArray array, TOther value, INDArray output)
		{
			CheckSameLength(array, output);

			IDataBuffer<float> _arrayData = ((NDArray<float>) array).data;
			IDataBuffer<float> _outputData = ((NDArray<float>) output).data;

			float _value = (float) System.Convert.ChangeType(value, typeof(float));

			for (long i = 0; i < _arrayData.Length; i++)
			{
				_outputData.SetValue(_arrayData.GetValue(i) + _value, i);
			}
		}

		public void Subtract<TOther>(INDArray array, TOther value, INDArray output)
		{
			CheckSameLength(array, output);

			IDataBuffer<float> _arrayData = ((NDArray<float>) array).data;
			IDataBuffer<float> _outputData = ((NDArray<float>) output).data;

			float _value = (float) System.Convert.ChangeType(value, typeof(float));

			for (long i = 0; i < _arrayData.Length; i++)
			{
				_outputData.SetValue(_arrayData.GetValue(i) - _value, i);
			}
		}

		public void Multiply<TOther>(INDArray array, TOther value, INDArray output)
		{
			CheckSameLength(array, output);

			IDataBuffer<float> _arrayData = ((NDArray<float>) array).data;
			IDataBuffer<float> _outputData = ((NDArray<float>) output).data;

			float _value = (float) System.Convert.ChangeType(value, typeof(float));

			for (long i = 0; i < _arrayData.Length; i++)
			{
				_outputData.SetValue(_arrayData.GetValue(i) * _value, i);
			}
		}

		public void Divide<TOther>(INDArray array, TOther value, INDArray output)
		{
			CheckSameLength(array, output);

			IDataBuffer<float> _arrayData = ((NDArray<float>) array).data;
			IDataBuffer<float> _outputData = ((NDArray<float>) output).data;

			float _value = (float) System.Convert.ChangeType(value, typeof(float));

			for (long i = 0; i < _arrayData.Length; i++)
			{
				_outputData.SetValue(_arrayData.GetValue(i) / _value, i);
			}
		}

		private void CheckSameLength(INDArray a, INDArray b)
		{
			if (a.Length != b.Length)
			{
				throw new ArgumentException($"NDArray lengths must be equal, but first array was of length {a.Length} and second {b.Length}.");
			}
		}
	}
}
