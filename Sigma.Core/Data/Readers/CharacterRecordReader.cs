/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A character-level reader, which reads records in characterwise.
	/// </summary>
	public class CharacterRecordReader : IRecordReader
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.</summary>
		public void Dispose()
		{
			throw new NotImplementedException();
		}

		/// <summary>
		/// The underlying data set source.
		/// </summary>
		public IDataSource Source { get; }

		/// <summary>
		/// Prepare this record reader and its underlying resources to be read.
		/// Note: This function may be called more than once (and subsequent calls should probably be ignored, depending on the implementation). 
		/// </summary>
		public void Prepare()
		{
			throw new NotImplementedException();
		}

		/// <summary>
		/// Reads a number of records in any format (therefore attached extractors must be compatible). 
		/// This method is mostly used internally by extractors, which have to be compatible with the used record readers (so they know which data format to expect). 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to read.</param>
		/// <returns>An object of the given type representing a collection of a given number of records.</returns>
		public object Read(int numberOfRecords)
		{
			throw new NotImplementedException();
		}

		/// <summary>
		/// Attach a certain record extractor to this record reader.
		/// </summary>
		/// <param name="extractor">The extractor to attach this reader to.</param>
		/// <returns>The given extractor (for convenience).</returns>
		public IRecordExtractor Extractor(IRecordExtractor extractor)
		{
			throw new NotImplementedException();
		}
	}
}
