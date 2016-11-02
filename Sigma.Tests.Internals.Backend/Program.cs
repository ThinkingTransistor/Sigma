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
