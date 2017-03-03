/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.IO;
using System.Runtime.Serialization;

namespace Sigma.Core.Persistence
{
	/// <summary>
	/// A utility class for serialising and de-serialising various objects to and from streams (e.g. file, network). 
	/// </summary>
	public static class Serialisation
	{
		/// <summary>
		/// Write an object to a target stream using a certain serialiser.
		/// </summary>
		/// <param name="obj">The object.</param>
		/// <param name="target">The target stream.</param>
		/// <param name="serialiser">The serialiser.</param>
		public static void Write(object obj, Stream target, ISerialiser serialiser)
		{
			serialiser.Write(obj, target);
		}

		/// <summary>
		/// Read an object from a target stream using a certain serialiser.
		/// </summary>
		/// <param name="target">The target stream.</param>
		/// <param name="serialiser">The serialiser.</param>
		/// <returns>The read object of the requested type.</returns>
		public static T Read<T>(Stream target, ISerialiser serialiser)
		{
			object read = serialiser.Read(target);

			if (!(read is T))
			{
				throw new SerializationException($"Unable to read object of type {typeof(T)} from target {target} using serialiser {serialiser}, read object {read} was not of the requested type.");
			}

			return (T) read;
		}
	}
}
