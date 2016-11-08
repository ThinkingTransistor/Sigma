/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using System;
using System.Collections.Generic;

namespace Sigma.Core.Data.Extractors
{
	/// <summary>
	/// A record extractor which extracts records from a record reader into an INDArray.
	/// Example: CSVRecordReader reads IDataSetSource as CSV data, CSVRecordExtractor extracts specific entries into an INDArray.
	/// 
	/// </summary>
	public interface IRecordExtractor
	{
		/// <summary>
		/// The underlying record reader that is attached to this extractor.
		/// </summary>
		IRecordReader Reader { get; set; }

		/// <summary>
		/// The names of all sections used in this extractor (e.g. "inputs" or "targets").
		/// These sections must be the same sections returned when calling Extract. 
		/// </summary>
		string[] SectionNames { get; }

		/// <summary>
		/// Extract a number of records into a collection of named ndarrays from the underlying reader. 
		/// The ndarray format to use is BatchTimeFeatures (BTF), where T is 1 for non-sequential data. 
		/// Note: There cannot be duplicate names (multiple ndarrays with the same name/identifier) in the final dataset. 
		///		  For merging multiple sources with the same name (e.g. inputs from different files) use merge extractors. 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <param name="handler">The computation handler to use for ndarray creation and manipulation.</param>
		/// <returns>The extracted named ndarrays, each containing a collection of numberOfRecords records (or less if unavailable).</returns>
		Dictionary<string, INDArray> Extract(int numberOfRecords, IComputationHandler handler);

		/// <summary>
		/// Extract a number of records from data read by any reader (requires the data formats to be compatible).
		/// The ndarray format to use is BatchTimeFeatures (BTF), where T is 1 for non-sequential data. 
		/// Note: There cannot be duplicate names (multiple ndarrays with the same name/identifier) in the final dataset. 
		///		  For merging multiple sources with the same name (e.g. inputs from different files) use merge extractors. 
		/// </summary>
		/// <param name="readData">The data read by any reader (requires the data formats to be compatible).</param>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <param name="handler">The computation handler to use for ndarray creation and manipulation.</param>
		/// <returns>The extracted named ndarrays, each containing a collection of numberOfRecords records (or less if unavailable).</returns>
		Dictionary<string, INDArray> ExtractFrom(object readData, int numberOfRecords, IComputationHandler handler);

		/// <summary>
		/// Prepare this record extractor and its underlying resources for extraction.
		/// </summary>
		void Prepare();

		/// <summary>
		/// Attach a number of preprocessors to this extractor. Preprocessors are applied in the order they were passed.
		/// </summary>
		/// <param name="preprocessors">The preprocessors to add to this extractor.</param>
		/// <returns>The last given preprocessor (for convenience).</returns>
		IRecordPreprocessor Preprocess(params IRecordPreprocessor[] preprocessors);
	}
}
