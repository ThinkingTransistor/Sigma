/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;
using System;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A record reader which reads a selected number of records from a data set source in a specific format (e.g. CSV).
	/// </summary>
	public interface IRecordReader : IDisposable
	{
		/// <summary>
		/// The underlying data set source.
		/// </summary>
		IDataSource Source { get; }

		/// <summary>
		/// Attach a certain record extractor to this record reader.
		/// </summary>
		/// <param name="extractor">The extractor to attach this reader to.</param>
		/// <returns>The given extractor (for convenience).</returns>
		IRecordExtractor Extractor(IRecordExtractor extractor);

		/// <summary>
		/// Prepare this record reader and its underlying resources to be read.
		/// Note: This function may be called more than once (and subsequent calls should probably be ignored, depending on the implementation). 
		/// </summary>
		void Prepare();

		/// <summary>
		/// Reads a number of records in any format (therefore attached extractors must be compatible). 
		/// This method is mostly used internally by extractors, which have to be compatible with the used record readers (so they know which data format to expect). 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to read.</param>
		/// <returns>An object of the given type representing a collection of a given number of records.</returns>
		object Read(int numberOfRecords);
	}
}
