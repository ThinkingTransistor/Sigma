/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.Serialization;
using log4net;

namespace Sigma.Core.Persistence
{
	/// <summary>
	/// A utility class for serialising and de-serialising various objects to and from streams (e.g. file, network). 
	/// </summary>
	public static class Serialisation
	{
		private static readonly ILog ClazzLogger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Write an object to a target stream using a certain serialiser.
		/// All related objects along the object graph are properly stored using the <see cref="ISerialisationNotifier"/> contract.
		/// </summary>
		/// <param name="obj">The object.</param>
		/// <param name="target">The target stream.</param>
		/// <param name="serialiser">The serialiser.</param>
		/// <param name="autoClose">Optionally indicate if the stream should be automatically closed.</param>
		public static void Write(object obj, Stream target, ISerialiser serialiser, bool autoClose = true)
		{
			if (obj == null) throw new ArgumentNullException(nameof(obj));
			if (target == null) throw new ArgumentNullException(nameof(target));
			if (serialiser == null) throw new ArgumentNullException(nameof(serialiser));

			ClazzLogger.Debug($"Writing object {obj} of type {obj.GetType()} to target stream {target} using serialiser {serialiser}...");

			Stopwatch stopwatch = Stopwatch.StartNew();
			long beforePosition = target.Position;

			TraverseObjectGraph(obj, new HashSet<object>(), (p, f, o) => (o as ISerialisationNotifier)?.OnSerialising());
			serialiser.Write(obj, target);
			TraverseObjectGraph(obj, new HashSet<object>(), (p, f, o) => (o as ISerialisationNotifier)?.OnSerialised());

			target.Flush();

			long bytesWritten = target.Position - beforePosition;

			target.Close();

			ClazzLogger.Debug($"Done writing object {obj} to target stream {target} using serialiser {serialiser}, " +
						  $"wrote {(bytesWritten / 1024.0):#.#}kB, took {stopwatch.ElapsedMilliseconds}ms.");
		}

		/// <summary>
		/// Read an object from a target stream using a certain serialiser.
		/// All related objects along the object graph are properly restored using the <see cref="ISerialisationNotifier"/> contract.
		/// </summary>
		/// <param name="target">The target stream.</param>
		/// <param name="serialiser">The serialiser.</param>
		/// <returns>The read object of the requested type.</returns>
		public static T Read<T>(Stream target, ISerialiser serialiser)
		{
			if (target == null) throw new ArgumentNullException(nameof(target));
			if (serialiser == null) throw new ArgumentNullException(nameof(serialiser));

			ClazzLogger.Debug($"Reading object from target stream {target} using serialiser {serialiser}...");

			Stopwatch stopwatch = Stopwatch.StartNew();
			long beforePosition = target.Position;

			object read = serialiser.Read(target);

			if (!(read is T))
			{
				throw new SerializationException($"Unable to read object from target {target} using serialiser {serialiser}, read object {read} was not of the requested type.");
			}

			TraverseObjectGraph(read, new HashSet<object>(), (parent, field, obj) =>
			{
				// automatically restore all logger instances
				if (field.FieldType == typeof(ILog))
				{
					field.SetValue(parent, LogManager.GetLogger(parent.GetType()));
				}

				(obj as ISerialisationNotifier)?.OnDeserialised();
			});

			ClazzLogger.Debug($"Done reading object {read} from target stream {target} using serialiser {serialiser}, " +
							  $"read {((target.Position - beforePosition) / 1024.0):#.#}kB, took {stopwatch.ElapsedMilliseconds}ms.");

			return (T) read;
		}

		internal static void TraverseObjectGraph(object root, ISet<object> traversedObjects, Action<object, FieldInfo, object> action)
		{
			Type type = root.GetType();
			traversedObjects.Add(root);

			do
			{
				FieldInfo[] fields = type.GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);

				foreach (FieldInfo field in fields)
				{
					object value = field.GetValue(root);

					if (value != null)
					{
						// TODO smarter optimisation than checking for system namespace, maybe build cache with type information?
						string ns = value.GetType().Namespace;
						bool boringSystemType = ns.StartsWith("System") && !ns.StartsWith("System.Collections");

						if (!boringSystemType && !traversedObjects.Contains(value))
						{
							TraverseObjectGraph(value, traversedObjects, action);

							// directly iterate over raw arrays since arrays don't have any fields containing the actual values
							//  and we still want to traverse all array members in our object graph
							Array valueAsArray = value as Array;

							if (valueAsArray != null)
							{
								foreach (object element in valueAsArray)
								{
									if (element != null && !traversedObjects.Contains(element))
									{
										TraverseObjectGraph(element, traversedObjects, action);
									}
								}
							}
						}
					}

					action.Invoke(root, field, value);
				}
			} while ((type = type.BaseType) != null && type != typeof(object));
		}
	}
}
