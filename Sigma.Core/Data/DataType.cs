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
	public interface IDataType
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
		Array CreateArray(int length);
	}

	/// <summary>
	/// A collection of common data type constants, as used by the internal data handling implementations.
	/// </summary>
	public class DataTypes
	{
		public static readonly IDataType FLOAT32 = new DataType<float>(4);
		public static readonly IDataType FLOAT64 = new DataType<double>(8);

		public static readonly IDataType INT8 = new DataType<byte>(1);
		public static readonly IDataType INT16 = new DataType<short>(2);
		public static readonly IDataType INT32 = new DataType<int>(4);
		public static readonly IDataType INT64 = new DataType<long>(8);
	}

	public class DataType<T> : IDataType
	{
		public int SizeBytes
		{
			get; private set;
		}

		public Type UnderlyingType
		{
			get;
		} = typeof(T);

		public Array CreateArray(int length)
		{
			return new T[length];
		}

		public DataType(int sizeBytes)
		{
			this.SizeBytes = sizeBytes;
		}
	}
}
