/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace Sigma.Core.Persistence
{
	/// <summary>
	/// An abstract serialiser interface for various serialisation formats (beyond C# Formatter variants).
	/// </summary>
	public interface ISerialiser
	{
		/// <summary>
		/// Write an object to a stream.
		/// </summary>
		/// <param name="obj">The object to write.</param>
		/// <param name="stream">The stream to write to.</param>
		void Write(object obj, Stream stream);

		/// <summary>
		/// Read an object from a stream.
		/// </summary>
		/// <param name="stream">The stream to read from..</param>
		/// <returns>The read object.</returns>
		object Read(Stream stream);
	}

	/// <summary>
	/// A collection of available serialisers.
	/// </summary>
	public static class Serialisers
	{
		/// <summary>
		/// A binary serialiser.
		/// </summary>
		public static readonly ISerialiser BinarySerialiser = new BinarySerialiser();
	}

	/// <summary>
	/// A binary serialiser, using the <see cref="BinaryFormatter"/>.
	/// </summary>
	public class BinarySerialiser : ISerialiser
	{
		public void Write(object obj, Stream stream)
		{
			new BinaryFormatter().Serialize(stream, obj);
		}

		public object Read(Stream stream)
		{
			return new BinaryFormatter().Deserialize(stream);
		}
	}
}
