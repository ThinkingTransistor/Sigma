using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data
{
	/// <summary>
	/// A data type that can be used for data buffers and mathematical operations. Data types are used to define buffer types to use at runtime.
	/// </summary>
	/// <typeparam name="T"></typeparam>
	public interface IDataType<T>
	{
		/// <summary>
		/// The underlying system type of this data type. 
		/// </summary>
		System.Type UnderlyingType { get; }

		/// <summary>
		/// The size of this type in bytes.
		/// </summary>
		int SizeBytes { get; }

		/// <summary>
		/// Creates a generic array with a given length and the underlying type of this data type as type. 
		/// </summary>
		/// <param name="length">The length of the array to create.</param>
		/// <returns>An array with the underlying type of this data type as type and the given length.</returns>
		T[] CreateArray(int length);
	}

	public class DataType<T> : IDataType<T>
	{
		public static readonly IDataType<double> FLOAT64 = new DataType<double>(8);
		public static readonly IDataType<float> FLOAT32 = new DataType<float>(4);

		public static readonly IDataType<byte> INT8 = new DataType<byte>(1);
		public static readonly IDataType<short> INT16 = new DataType<short>(2);
		public static readonly IDataType<int> INT32 = new DataType<int>(4);
		public static readonly IDataType<long> INT64 = new DataType<long>(8);

		public int SizeBytes
		{
			get
			{
				return SizeBytes;
			}

			private set
			{
				SizeBytes = value;
			}
		}

		public System.Type UnderlyingType
		{
			get;
		} = typeof(T);

		public T[] CreateArray(int length)
		{
			return new T[length];
		}

		public DataType(int sizeBytes)
		{
			this.SizeBytes = sizeBytes;
		}
	}
}
