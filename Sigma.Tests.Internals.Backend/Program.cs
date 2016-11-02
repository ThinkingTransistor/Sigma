using Sigma.Core.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
using System.Diagnostics;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			long length = 400000L;
			long offset = 100L;
			DataBuffer<double> rootBuffer = new DataBuffer<double>(length);
			DataBuffer<double> childBufferL2 = new DataBuffer<double>(rootBuffer, offset, length - offset * 2);
			DataBuffer<double> childBufferL3 = new DataBuffer<double>(childBufferL2, offset, length - offset * 4);

			rootBuffer.SetValues(new double[] { 0.0, 1.1, 2.2, 3.3, 4.4, 5.5 }, 1, 200, 4);
			childBufferL2.SetValues(new double[] { 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1 }, 1, 104, 3);
			childBufferL3.SetValues(new double[] { 8.8, 9.9, 10.1, 11.11, 12.12, 13.13 }, 0, 7, 3);

			Debug.WriteLine(childBufferL3.GetValuesArray(0, 10).GetValuesPackedArray(0, 10));
		}

		public interface INDArray
		{

		}

		public interface IHandler
		{
			INDArray Add(INDArray a, INDArray b);
		}

		public class CPUFloat32Handler : IHandler
		{
			public IDataType DataType { get; }

			public CPUFloat32Handler()
			{
				this.DataType = DataTypes.FLOAT32;
			}

			public INDArray Create(int[] shape)
			{
				return null;
			}

			public INDArray Add(INDArray a, INDArray b)
			{
				NDArray<float> _a = (NDArray<float>) a;
				NDArray<float> _b = (NDArray<float>) b;
				NDArray<float> result = _b; //CreateNDArray(...);

				for (int i = 0; i < _a.data.Length; i++)
				{
					result.data.SetValue(_a.data.GetValue(i) + _b.data.GetValue(i), i);
				}

				return result;
			}
		}

		public class NDArray<T> : INDArray
		{
			public IDataBuffer<T> data;
			public int[] shape;

			public T GetValue(int[] indices)
			{
				return default(T);
				//return data.GetValue(IndiciesMagic());
			}
		}
	}
}
