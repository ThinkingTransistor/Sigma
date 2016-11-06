/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Sources;

namespace Sigma.Core.Data.Readers
{
	/// <summary>
	/// A record reader which reads a selected number of records from a data set source in a specific format (e.g. CSV).
	/// </summary>
	public interface IRecordReader
	{
		/// <summary>
		/// The underlying data set source.
		/// </summary>
		IDataSetSource Source { get; }

		/// <summary>
		/// Attach a certain record extractor to this record reader.
		/// </summary>
		/// <param name="extractor">The extractor to attach this reader to.</param>
		/// <returns>The given extractor (for convenience).</returns>
		IRecordExtractor Extractor(IRecordExtractor extractor);

		/// <summary>
		/// Prepare this record reader and its underlying resources to be read.
		/// </summary>
		void Prepare();

		/// <summary>
		/// Reads a number of records and converts them to the specified type. 
		/// This method is mostly used internally by extractors, which have to be compatible with the used record readers (so they know which data format to expect). 
		/// </summary>
		/// <typeparam name="T">The type the collection of records have.</typeparam>
		/// <param name="numberOfRecords">The number of records to read.</param>
		/// <returns>An object of the given type representing a collection of a given number of records.</returns>
		T Read<T>(int numberOfRecords);
	}
}
