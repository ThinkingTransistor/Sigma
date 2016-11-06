/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.IO;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A file resource on the local file system used for datasets.
	/// </summary>
	public class FileSource : IDataSetSource
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private bool exists;
		private string fullPath;
		private string localPath;

		private Stream fileStream;

		public FileSource(string path) : this(path, SigmaEnvironment.Globals.Get<string>("datasets"))
		{
		}

		public FileSource(string path, string datasetsPath)
		{
			if (path == null)
			{
				throw new ArgumentNullException("Path cannot be null.");
			}

			if (datasetsPath == null)
			{
				throw new ArgumentNullException("Data sets path cannot be null (are the SigmaEnironment.Globals missing?)");
			}

			this.localPath = path;
			this.fullPath = datasetsPath + path;

			CheckExists();
		}

		private bool CheckExists()
		{
			return this.exists = File.Exists(fullPath);
		}

		public bool Chunkable
		{
			get { return true; }
		}

		public bool Exists()
		{
			return exists;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare file source, underlying file \"{fullPath}\" does not exist.");
			}

			if (fileStream == null)
			{
				fileStream = new FileStream(fullPath, FileMode.Open);

				logger.Info($"Opened file \"{fullPath}\".");
			}
		}

		public Stream Retrieve()
		{
			if (fileStream == null)
			{
				throw new InvalidOperationException("Cannot retrieve file source, file stream was not established (missing or failed Prepare() call?).");
			}

			return fileStream;
		}

		public void Dispose()
		{
			this.fileStream?.Dispose();
		}
	}
}
