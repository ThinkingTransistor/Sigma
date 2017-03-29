/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using System;
using System.IO;
using Sigma.Core.Persistence;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A file resource on the local file system used for datasets.
	/// </summary>
	[Serializable]
	public class FileSource : IDataSource, ISerialisationNotifier
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public string ResourceName { get; }

		private bool _exists;
		private readonly string _fullPath;
		private long _internalStreamPositionBytes;

		[NonSerialized]
		private FileStream _fileStream;

		/// <summary>
		/// Create a file source referring to a locally stored file.
		/// </summary>
		/// <param name="path">The local fileName (relative to the global dataset fileName).</param>
		public FileSource(string path) : this(path, SigmaEnvironment.Globals.Get<string>("datasets_path"))
		{
		}

		/// <summary>
		/// Create a file source referring to a locally stored file.
		/// </summary>
		/// <param name="fileName">The file name of this file source.</param>
		/// <param name="directory">The directory, within which the local fileName will be stored.</param>
		public FileSource(string fileName, string directory)
		{
			if (fileName == null) throw new ArgumentNullException(nameof(fileName));
			if (directory == null) throw new ArgumentNullException(nameof(directory));

			// sanitise possible inconsistent directory names (ours end with / but some might not and people are lazy) 
			directory = directory.Replace('\\', '/');
			if (!directory.EndsWith("/"))
			{
				directory = directory + "/";
			}

			_fullPath = directory + fileName;

			ResourceName = new FileInfo(fileName).Name;

			CheckExists();
		}

		/// <summary>
		/// Called before this object is serialised.
		/// </summary>
		public void OnSerialising()
		{
			if (_fileStream != null)
			{
				_internalStreamPositionBytes = _fileStream.Position;
			}
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
			if (_fileStream != null)
			{
				_logger.Debug($"Restoring file stream state after deserialisation...");

				Prepare();

				_fileStream.Seek(_internalStreamPositionBytes, SeekOrigin.Begin);

				_logger.Debug($"Done restoring file stream state.");
			}
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

				_logger.Debug($"Opened file \"{_fullPath}\".");
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
