/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using ICSharpCode.SharpZipLib.BZip2;
using ICSharpCode.SharpZipLib.LZW;
using ICSharpCode.SharpZipLib.Tar;
using ICSharpCode.SharpZipLib.Zip;
using log4net;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// An unpacker which takes an input stream and decompresses (unpacks) it using a certain compression method. Used for compressed sources.
	/// </summary>
	public interface IUnpacker
	{
		/// <summary>
		/// Unpack a certain stream with the decompression algorithm implemented in this unpacker. 
		/// </summary>
		/// <param name="input">The stream to unpack.</param>
		/// <returns>A stream with the unpacked contents of the given input stream.</returns>
		Stream Unpack(Stream input);
	}

	/// <summary>
	/// A collection of the by default supported decompression algorithms and an extension-unpacker registry for inferring algorithms.
	/// </summary>
	public static class Unpackers
	{
		private static readonly Dictionary<string, IUnpacker> RegisteredUnpackersByExtension = new Dictionary<string, IUnpacker>();
		private static readonly ILog Logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public static readonly IUnpacker GzipUnpacker = Register(".gz", new GZipUnpacker());
		public static readonly IUnpacker TarUnpacker = Register(".tar", new TarUnpacker());
		public static readonly IUnpacker ZipUnpacker = Register(".zip", new ZipUnpacker());
		public static readonly IUnpacker Bzip2Unpacker = Register(".bz2", new BZip2Unpacker());
		public static readonly IUnpacker LzwUnpacker = Register(".z", new LzwUnpacker());

		public static bool AllowExternalTypeOverwrites { get; set; } = false;

		public static IUnpacker Register(string extension, IUnpacker unpacker)
		{
			extension = extension.ToLower();

			if (!RegisteredUnpackersByExtension.ContainsKey(extension))
			{
				RegisteredUnpackersByExtension.Add(extension, unpacker);
			}
			else
			{
				if (AllowExternalTypeOverwrites)
				{
					Logger.Info($"Overwrote internal resource extension {extension} to now refer to the unpacker {unpacker} (this may not be what you wanted).");
				}
				else
				{
					throw new ArgumentException($"Extension {extension} is already registered as {RegisteredUnpackersByExtension[extension]} and cannot be changed to {unpacker} (AllowExternalTypeOverwrites flag is set to false).");
				}
			}

			return unpacker;
		}

		public static IUnpacker GetMatchingUnpacker(string extension)
		{
			extension = extension.ToLower();

			if (!RegisteredUnpackersByExtension.ContainsKey(extension))
			{
				Logger.Warn($"There is no extension-unpacker mapping for {extension} in the internal extension-unpacker registry.");

				return null;
			}

			return RegisteredUnpackersByExtension[extension];
		}
	}

	/// <summary>
	/// A GZip unpacker using the default Systems.IO.Compression GZipStream implementation.
	/// </summary>
	[Serializable]
	public class GZipUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new GZipStream(input, CompressionMode.Decompress);
		}
	}

	/// <summary>
	/// A Tar unpacker using the SharpZipLib TarInputStream implementation.
	/// </summary>
	[Serializable]
	public class TarUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new TarInputStream(input);
		}
	}

	/// <summary>
	/// A Zip unpacker using the SharpZipLib ZipInputStream implementation.
	/// </summary>
	[Serializable]
	public class ZipUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new ZipInputStream(input);
		}
	}

	/// <summary>
	/// A BZip2 unpacker using the SharpZipLib BZip2Unpacker implementation.
	/// </summary>
	[Serializable]
	public class BZip2Unpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new BZip2InputStream(input);
		}
	}

	/// <summary>
	/// A LZW unpacker using the SharpZipLib LzwInputStream implementation.
	/// </summary>
	[Serializable]
	public class LzwUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new LzwInputStream(input);
		}
	}
}
