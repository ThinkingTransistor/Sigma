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
	/// A compressed data set source. Decompresses a given underlying source using a given or inferred unpacker.
	/// During preparation the entire stream is decompressed and stored as a local file.
	/// </summary>
	public class CompressedSource : IDataSource
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public bool Seekable => true;

		public string ResourceName => UnderlyingSource?.ResourceName;

		/// <summary>
		/// The underlying data source which is decompressed.
		/// </summary>
		public IDataSource UnderlyingSource { get; }


		/// <summary>
		/// The unpacker to use to decompress the underlying data source.
		/// </summary>
		public IUnpacker Unpacker { get; }

		private FileStream _fileStream;
		private readonly string _localUnpackPath;
		private bool _prepared;

		/// <summary>
		/// Create a compressed source which automatically decompressed a certain underlying source. 
		/// The decompression algorithm to use and the local unpacked path are inferred.
		/// </summary>
		/// <param name="source">The underlying data set source to decompress.</param>
		public CompressedSource(IDataSource source) : this(source, InferLocalUnpackPath(source))
		{
		}

		/// <summary>
		/// Create a compressed source which automatically decompressed a certain underlying source. 
		/// The decompression algorithm to use is inferred, the local unpacked path is explicitly given.
		/// </summary>
		/// <param name="source">The underlying data set source to decompress.</param>
		/// <param name="localUnpackPath">The local unpack path (where the unpacked underlying source stream is stored locally).</param>
		public CompressedSource(IDataSource source, string localUnpackPath) : this(source, localUnpackPath, InferUnpacker(source))
		{
		}

		/// <summary>
		/// Create a compressed source which automatically decompressed a certain underlying source. 
		/// The decompression algorithm to use and the local unpacked path are explicitly given.
		/// </summary>
		/// <param name="source">The underlying data set source to decompress.</param>
		/// <param name="localUnpackPath">The local unpack path (where the unpacked underlying source stream is stored locally).</param>
		/// <param name="unpacker">The unpacker to use to decompress the given source.</param>
		public CompressedSource(IDataSource source, string localUnpackPath, IUnpacker unpacker)
		{
			if (source == null)
			{
				throw new ArgumentNullException(nameof(source));
			}

			if (localUnpackPath == null)
			{
				throw new ArgumentNullException(nameof(localUnpackPath));
			}

			if (unpacker == null)
			{
				throw new ArgumentNullException(nameof(unpacker));
			}

			UnderlyingSource = source;
			_localUnpackPath = localUnpackPath;
			Unpacker = unpacker;
		}

		public bool Exists()
		{
			return UnderlyingSource.Exists();
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare compressed source, underlying source \"{UnderlyingSource}\" does not exist.");
			}

			if (!_prepared)
			{
				UnderlyingSource.Prepare();

				DirectoryInfo directoryInfo = new FileInfo(_localUnpackPath).Directory;
				if (directoryInfo != null)
				{
					Directory.CreateDirectory(directoryInfo.FullName);
				}

				Stream sourceStream = UnderlyingSource.Retrieve();

				_logger.Info($"Unpacking source stream using unpacker {Unpacker} to local unpack path \"{_localUnpackPath}\"...");

				Stream unpackedStream = Unpacker.Unpack(sourceStream);

				FileStream decompressedFileStream = new FileStream(_localUnpackPath, FileMode.OpenOrCreate);

				unpackedStream.CopyTo(decompressedFileStream);

				_logger.Info($"Done unpacking source stream using unpacker {Unpacker} to local unpack path \"{_localUnpackPath}\" (unpacked size {decompressedFileStream.Length / 1024L}kB).");

				decompressedFileStream.Close();

				_fileStream = new FileStream(_localUnpackPath, FileMode.Open);

				_prepared = true;
			}
		}

		public Stream Retrieve()
		{
			if (!_prepared)
			{
				throw new InvalidOperationException("Cannot retrieve compressed source, compressed source was not prepared correctly (missing or failed Prepare() call?).");
			}

			return _fileStream;
		}

		public void Dispose()
		{
			UnderlyingSource.Dispose();
		}

		private static string InferLocalUnpackPath(IDataSource source)
		{
			return SigmaEnvironment.Globals["datasets"] + Path.GetFileNameWithoutExtension(source.ResourceName);
		}

		private static IUnpacker InferUnpacker(IDataSource source)
		{
			string resourceName = source.ResourceName;
			string extension = Path.HasExtension(resourceName) ? Path.GetExtension(resourceName) : null;

			if (extension == null || extension.Length == 0)
			{
				throw new ArgumentException($"Unable to infer unpacker via extension for underlying source {source} with resource name {source.ResourceName}, extension could not be identified.");
			}
			else
			{
				IUnpacker inferredUnpacker = Unpackers.GetMatchingUnpacker(extension);

				if (inferredUnpacker == null)
				{
					throw new ArgumentException($"Unable to infer unpacker via extension {extension} of resource name {source.ResourceName} with internal registry.");
				}
				else
				{
					return inferredUnpacker;
				}
			}
		}
	}
}
