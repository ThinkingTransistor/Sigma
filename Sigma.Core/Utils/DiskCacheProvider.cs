/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using Sigma.Core.Persistence;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A cache provider implementation that uses a directory on the local disk. 
	/// </summary>
	[Serializable]
	public class DiskCacheProvider : ICacheProvider, ISerialisationNotifier
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string RootDirectory { get; }
		public string CacheFileExtension { get; set; } = ".cache";

		[NonSerialized]
		private IFormatter _serialisationFormatter;

		/// <summary>
		/// Create a disk cache provider with a certain root directory.
		/// </summary>
		/// <param name="rootDirectory">The root directory.</param>
		public DiskCacheProvider(string rootDirectory)
		{
			if (rootDirectory == null)
			{
				throw new ArgumentNullException(nameof(rootDirectory));
			}

			if (!Directory.Exists(rootDirectory))
			{
				Directory.CreateDirectory(rootDirectory);
			}

			if (!rootDirectory.EndsWith("/") && !rootDirectory.EndsWith("\\"))
			{
				rootDirectory = rootDirectory + (rootDirectory.Contains("/") ? '/' : '\\'); 
			}

			RootDirectory = rootDirectory;
			_serialisationFormatter = new BinaryFormatter();
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
		}

		/// <summary>
		/// Called after this object was serialised.
		/// </summary>
		public void OnSerialised()
		{
		}

		/// <summary>
		/// Called after this object was de-serialised. 
		/// </summary>
		public void OnDeserialised()
		{
			_serialisationFormatter = new BinaryFormatter();
		}

		public bool IsCached(string identifier)
		{
			return File.Exists(RootDirectory + identifier + CacheFileExtension);
		}

		public void Store(string identifier, object data)
		{
			ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Save, $"storing {identifier} on disk", indeterminate: true);

			_logger.Debug($"Caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Create);
			}

			using (fileStream)
			{
				Serialisation.Write(data, fileStream, Serialisers.BinarySerialiser);
			}

			_logger.Debug($"Done caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\".");

			SigmaEnvironment.TaskManager.EndTask(task);
		}

		public T Load<T>(string identifier)
		{
			if (!IsCached(identifier))
			{
				return default(T);
			}

			ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Load, $"loading {identifier} from disk", indeterminate: true);

			_logger.Debug($"Loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Open);
			}

			using (fileStream)
			{
				try
				{
					T obj = Serialisation.Read<T>(fileStream, Serialisers.BinarySerialiser);

					_logger.Debug($"Done loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");

					SigmaEnvironment.TaskManager.EndTask(task);

					return obj;
				}
				catch (Exception e)
				{
					_logger.Warn($"Failed to load cache entry for identifier \"{identifier}\" with error \"{e.GetType()}\", returning default value for type.");
					_logger.Debug(e);

					SigmaEnvironment.TaskManager.CancelTask(task);

					return default(T);
				}
			}
		}

		public void Remove(string identifier)
		{
			if (IsCached(identifier))
			{
				_logger.Debug($"Removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

				lock (this)
				{
					File.Delete(RootDirectory + identifier);
				}

				_logger.Debug($"Done removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");
			}
		}

		public void RemoveAll()
		{
			string[] cacheFiles = Directory.GetFiles(RootDirectory, $"*{CacheFileExtension}", SearchOption.AllDirectories);

			_logger.Debug($"Removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\"...");

			lock (this)
			{
				foreach (string file in cacheFiles)
				{
					File.Delete(file);
				}
			}

			_logger.Debug($"Done removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\".");
		}

		public void Dispose()
		{
		}
	}
}
