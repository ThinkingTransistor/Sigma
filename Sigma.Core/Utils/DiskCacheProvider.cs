/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A cache provider implementation that uses a directory on the local disk. 
	/// </summary>
	public class DiskCacheProvider : ICacheProvider
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string RootDirectory { get; private set; }

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
			return File.Exists(RootDirectory + identifier);
		}

		public void Store(string identifier, object data)
		{
			logger.Info($"Caching object {data} with identifier {identifier} to disk to \"{RootDirectory + identifier}\"...");

			using (Stream fileStream = new FileStream(RootDirectory + identifier, FileMode.Create))
			{
				serialisationFormatter.Serialize(fileStream, data);
			}

			logger.Info($"Done caching object {data} with identifier {identifier} to disk to \"{RootDirectory + identifier}\".");
		}

		public T Load<T>(string identifier)
		{
			if (!IsCached(identifier))
			{
				return default(T);
			}

			using (Stream fileStream = new FileStream(RootDirectory + identifier, FileMode.Create))
			{
				try
				{
					return (T) serialisationFormatter.Deserialize(fileStream);
				}
				catch (Exception e)
				{
					logger.Warn($"Failed to load cache entry for identifier {identifier} with error {e}, returning default value for type.");

					return default(T);
				}
			}
		}

		public void Remove(string identifier)
		{
			if (IsCached(identifier))
			{
				File.Delete(RootDirectory + identifier);
			}
		}
	}
}
