/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

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
	public class FileSource : IDataSource
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string ResourceName { get; }

		private bool _exists;
		private readonly string _fullPath;

		private Stream _fileStream;

		/// <summary>
		/// Create a file source referring to a locally stored file.
		/// </summary>
		/// <param name="path">The local fileName (relative to the global dataset fileName).</param>
		public FileSource(string path) : this(path, SigmaEnvironment.Globals.Get<string>("datasets"))
		{
		}

		/// <summary>
		/// Create a file source referring to a locally stored file.
		/// </summary>
		/// <param name="fileName">The file name of this file source.</param>
		/// <param name="datasetsPath">The data set fileName, within which the local fileName will be stored.</param>
		public FileSource(string fileName, string datasetsPath)
		{
			if (fileName == null)
			{
				throw new ArgumentNullException(nameof(fileName));
			}

			if (datasetsPath == null)
			{
				throw new ArgumentNullException(nameof(datasetsPath));
			}

			_fullPath = datasetsPath + fileName;

			ResourceName = new FileInfo(fileName).Name;

			CheckExists();
		}

		private void CheckExists()
		{
			_exists = File.Exists(_fullPath);
		}

		public bool Seekable => true;

		public bool Exists()
		{
			return _exists;
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare file source, underlying file \"{_fullPath}\" does not exist.");
			}

			if (_fileStream == null)
			{
				_fileStream = new FileStream(_fullPath, FileMode.Open);

				_logger.Info($"Opened file \"{_fullPath}\".");
			}
		}

		public Stream Retrieve()
		{
			if (_fileStream == null)
			{
				throw new InvalidOperationException("Cannot retrieve file source, file stream was not established (missing or failed Prepare() call?).");
			}

			return _fileStream;
		}

		public virtual void Dispose()
		{
			_fileStream?.Dispose();
		}
	}
}
