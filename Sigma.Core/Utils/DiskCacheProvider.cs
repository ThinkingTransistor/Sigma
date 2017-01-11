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

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A cache provider implementation that uses a directory on the local disk. 
	/// </summary>
	public class DiskCacheProvider : ICacheProvider
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string RootDirectory { get; }
		public string CacheFileExtension { get; set; } = ".cache";

		private readonly IFormatter _serialisationFormatter;

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

		public bool IsCached(string identifier)
		{
			return File.Exists(RootDirectory + identifier + CacheFileExtension);
		}

		public void Store(string identifier, object data)
		{
			ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Save, $"storing {identifier} on disk", indeterminate: true);

			_logger.Info($"Caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Create);
			}

			using (fileStream)
			{
				_serialisationFormatter.Serialize(fileStream, data);
			}

			_logger.Info($"Done caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\".");

			SigmaEnvironment.TaskManager.EndTask(task);
		}

		public T Load<T>(string identifier)
		{
			if (!IsCached(identifier))
			{
				return default(T);
			}

			ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Load, $"loading {identifier} from disk", indeterminate: true);

			_logger.Info($"Loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Open);
			}

			using (fileStream)
			{
				try
				{
					T obj = (T) _serialisationFormatter.Deserialize(fileStream);

					_logger.Info($"Done loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");

					SigmaEnvironment.TaskManager.EndTask(task);

					return obj;
				}
				catch (Exception e)
				{
					_logger.Warn($"Failed to load cache entry for identifier \"{identifier}\" with error \"{e}\", returning default value for type.");

					SigmaEnvironment.TaskManager.CancelTask(task);

					return default(T);
				}
			}
		}

		public void Remove(string identifier)
		{
			if (IsCached(identifier))
			{
				_logger.Info($"Removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

				lock (this)
				{
					File.Delete(RootDirectory + identifier);
				}

				_logger.Info($"Done removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");
			}
		}

		public void RemoveAll()
		{
			string[] cacheFiles = Directory.GetFiles(RootDirectory, $"*{CacheFileExtension}", SearchOption.AllDirectories);

			_logger.Info($"Removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\"...");

			lock (this)
			{
				foreach (string file in cacheFiles)
				{
					File.Delete(file);
				}
			}

			_logger.Info($"Done removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\".");
		}

		public void Dispose()
		{
		}
	}
}
