/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

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
			using (Stream fileStream = new FileStream(RootDirectory + identifier, FileMode.Create))
			{
				serialisationFormatter.Serialize(fileStream, data);
			}
		}

		public T Load<T>(string identifier)
		{
			if (!IsCached(identifier))
			{
				return default(T);
			}

			using (Stream fileStream = new FileStream(RootDirectory + identifier, FileMode.Create))
			{
				return (T) serialisationFormatter.Deserialize(fileStream);
			}
		}
	}
}
