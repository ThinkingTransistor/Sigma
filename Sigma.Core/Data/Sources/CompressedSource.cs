/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using SharpCompress.Readers;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Sources
{
	public class CompressedSource : IDataSetSource
	{
		private static ILog clazzLogger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public bool Chunkable { get { return true; } }

		public string ResourceName { get { return UnderlyingSource?.ResourceName; } }

		public IDataSetSource UnderlyingSource { get; private set; }

		public IUnpacker Unpacker { get; private set; }

		private FileStream fileStream;
		private string localUnpackPath;
		private bool prepared;

		public CompressedSource(IDataSetSource source) : this(source, InferLocalUnpackPath(source))
		{
		}

		public CompressedSource(IDataSetSource source, string localUnpackPath) : this(source, localUnpackPath, InferUnpacker(source))
		{
		}

		public CompressedSource(IDataSetSource source, string localUnpackPath, IUnpacker unpacker)
		{
			if (source == null)
			{
				throw new ArgumentNullException("Source cannot be null.");
			}

			if (localUnpackPath == null)
			{
				throw new ArgumentNullException("Local unpack path cannot be null.");
			}

			if (unpacker == null)
			{
				throw new ArgumentNullException("Unpacker cannot be null.");
			}

			this.UnderlyingSource = source;
			this.localUnpackPath = localUnpackPath;
			this.Unpacker = unpacker;
		}

		public bool Exists()
		{
			return this.UnderlyingSource.Exists();
		}

		public void Prepare()
		{
			if (!Exists())
			{
				throw new InvalidOperationException($"Cannot prepare compressed source, underlying source \"{UnderlyingSource}\" does not exist.");
			}

			if (!prepared)
			{
				this.UnderlyingSource.Prepare();

				Directory.CreateDirectory(new FileInfo(localUnpackPath).Directory.FullName);

				Stream sourceStream = this.UnderlyingSource.Retrieve();

				logger.Info($"Unpacking source stream using unpacker {Unpacker} to local unpack path \"{localUnpackPath}\"...");

				Stream unpackedStream = this.Unpacker.Unpack(sourceStream);

				FileStream decompressedFileStream = new FileStream(localUnpackPath, FileMode.OpenOrCreate);

				unpackedStream.CopyTo(decompressedFileStream);

				logger.Info($"Done unpacking source stream using unpacker {Unpacker} to local unpack path \"{localUnpackPath}\" (unpacked size {decompressedFileStream.Length / 1024L}kB).");

				decompressedFileStream.Close();

				this.fileStream = new FileStream(localUnpackPath, FileMode.Open);

				prepared = true;
			}
		}

		public Stream Retrieve()
		{
			if (!prepared)
			{
				throw new InvalidOperationException("Cannot retrieve compressed source, compressed source was not prepared correctly (missing or failed Prepare() call?).");
			}

			return this.fileStream;
		}

		public void Dispose()
		{
			this.UnderlyingSource.Dispose();
		}

		private static string InferLocalUnpackPath(IDataSetSource source)
		{
			return SigmaEnvironment.Globals["datasets"] + Path.GetFileNameWithoutExtension(source.ResourceName);
		}

		private static IUnpacker InferUnpacker(IDataSetSource source)
		{
			string resourceName = source.ResourceName;
			string extension = Path.HasExtension(resourceName) ? Path.GetExtension(resourceName) : null;

			if (extension == null || extension.Length == 0)
			{
				clazzLogger.Info($"Unable to infer unpacker via extension for underlying source {source} with resource name {source.ResourceName}, will attempt to infer unpacker type from source stream signature when first retrieved.");
			}
			else
			{
				IUnpacker inferredUnpacker = Unpackers.GetMatchingUnpacker(extension);

				if (inferredUnpacker == null)
				{
					clazzLogger.Info($"Unable to infer unpacker via extension {extension} with internal registry, will attempt to infer unpacker type from source stream signature when first retrieved.");
				}
				else
				{
					return inferredUnpacker;
				}
			}

			return new SignatureDetectingUnpacker();
		}
	}
}
