/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

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
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string RootDirectory { get; private set; }
		public string CacheFileExtension { get; set; } = ".cache";

		private IFormatter serialisationFormatter;

		public DiskCacheProvider(string rootDirectory)
		{
			if (rootDirectory == null)
			{
				throw new ArgumentNullException("Root directory cannot be null.");
			}

			rootDirectory = rootDirectory.Replace('\\', '/');

			if (rootDirectory.EndsWith("/"))
			{
				rootDirectory = rootDirectory.Substring(0, rootDirectory.Length - 1);
			}

			if (!Directory.Exists(rootDirectory))
			{
				Directory.CreateDirectory(rootDirectory);
			}

			this.RootDirectory = rootDirectory + "/";
			this.serialisationFormatter = new BinaryFormatter();
		}

		public bool IsCached(string identifier)
		{
			return File.Exists(RootDirectory + identifier + CacheFileExtension);
		}

		public void Store(string identifier, object data)
		{
			logger.Info($"Caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Create);
			}

			using (fileStream)
			{
				serialisationFormatter.Serialize(fileStream, data);
			}

			logger.Info($"Done caching object {data} with identifier \"{identifier}\" to disk to \"{RootDirectory + identifier}\".");
		}

		public T Load<T>(string identifier)
		{
			if (!IsCached(identifier))
			{
				return default(T);
			}

			logger.Info($"Loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

			Stream fileStream;

			lock (this)
			{
				fileStream = new FileStream(RootDirectory + identifier + CacheFileExtension, FileMode.Open);
			}

			using (fileStream)
			{
				try
				{
					T obj = (T) serialisationFormatter.Deserialize(fileStream);

					logger.Info($"Done loading cache object with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");

					return obj;
				}
				catch (Exception e)
				{
					logger.Warn($"Failed to load cache entry for identifier \"{identifier}\" with error \"{e}\", returning default value for type.");

					return default(T);
				}
			}
		}

		public void Remove(string identifier)
		{
			if (IsCached(identifier))
			{
				logger.Info($"Removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\"...");

				lock (this)
				{
					File.Delete(RootDirectory + identifier);
				}

				logger.Info($"Done removing cache entry with identifier \"{identifier}\" from disk \"{RootDirectory + identifier + CacheFileExtension}\".");
			}
		}

		public void RemoveAll()
		{
			string[] cacheFiles = Directory.GetFiles(RootDirectory, $"*{CacheFileExtension}", SearchOption.AllDirectories);

			logger.Info($"Removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\"...");

			lock (this)
			{
				foreach (string file in cacheFiles)
				{
					File.Delete(file);
				}
			}

			logger.Info($"Done removing ALL of {cacheFiles.Length} cache entries from this provider from disk using pattern \"{RootDirectory}*{CacheFileExtension}\".");
		}

		public void Dispose()
		{
		}
	}
}
