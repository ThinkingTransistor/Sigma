using ICSharpCode.SharpZipLib.Tar;
using log4net;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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

	public static class Unpackers
	{
		private static Dictionary<string, IUnpacker> registeredUnpackersByExtension = new Dictionary<string, IUnpacker>();
		private static ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public static readonly IUnpacker GZIP_UNPACKER = Register(".gz", new GZipUnpacker());

		public static bool AllowExternalTypeOverwrites { get; set; } = false;

		public static IUnpacker Register(string extension, IUnpacker unpacker)
		{
			extension = extension.ToLower();

			if (!registeredUnpackersByExtension.ContainsKey(extension))
			{
				registeredUnpackersByExtension.Add(extension, unpacker);
			}
			else
			{
				if (AllowExternalTypeOverwrites)
				{
					logger.Info($"Overwrote internal resource extension {extension} to now refer to the unpacker {unpacker} (this may not be what you wanted).");
				}
				else
				{
					throw new ArgumentException($"Extension {extension} is already registered as {registeredUnpackersByExtension[extension]} and cannot be changed to {unpacker} (AllowExternalTypeOverwrites flag is set to false).");
				}
			}

			return unpacker;
		}

		public static IUnpacker GetMatchingUnpacker(string extension)
		{
			extension = extension.ToLower();

			if (!registeredUnpackersByExtension.ContainsKey(extension))
			{
				logger.Info($"There is no extension-unpacker mapping for {extension} in the internal extension-unpacker registry.");

				return null;
			}

			return registeredUnpackersByExtension[extension];
		}
	}

	public class GZipUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new GZipStream(input, CompressionMode.Decompress);
		}
	}

	public class TarUnpacker : IUnpacker
	{
		public Stream Unpack(Stream input)
		{
			return new TarInputStream(input);
		}
	}
}
