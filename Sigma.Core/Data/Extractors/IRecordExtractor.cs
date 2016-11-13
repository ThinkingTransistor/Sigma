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
	public interface IRecordExtractor : IDisposable
	{
		/// <summary>
		/// The underlying record reader that is attached to this extractor.
		/// </summary>
		IRecordReader Reader { get; set; }

		/// <summary>
		/// The underlying record extractor that may be attached to this extractor.
		/// If attached, this extractor will be used instead of the reader (which is expected to have the same underlying reader).
		/// </summary>
		IRecordExtractor ParentExtractor { get; set; }

		/// <summary>
		/// The names of all sections output in this extractor (e.g. "inputs" or "targets"). 
		/// These sections must be the same sections returned when calling any of the Extract methods. 
		/// </summary>
		string[] SectionNames { get; set; }

		/// <summary>
		/// Extract a number of records into a collection of named ndarrays directly from the underlying reader/extractor. 
		/// The ndarray format to use is BatchTimeFeatures (BTF), where T is 1 for non-sequential data. 
		/// Note: There cannot be duplicate names (multiple ndarrays with the same name/identifier) in the final dataset. 
		///		  For merging multiple sources with the same name (e.g. inputs from different files) use merge extractors. 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <param name="handler">The computation handler to use for ndarray creation and manipulation.</param>
		/// <returns>The extracted named ndarrays, each containing a collection of numberOfRecords records (or less if unavailable). Return null when not a single record could be extracted.</returns>
		Dictionary<string, INDArray> ExtractDirect(int numberOfRecords, IComputationHandler handler);

		/// <summary>
		/// Extract a number of records from data read or extracted by any other source (requires the data formats to be compatible).
		/// Direct extraction does not respect parent extractors and extracts directly from this extractor.
		/// The ndarray format to use is BatchTimeFeatures (BTF), where T is 1 for non-sequential data. 
		/// Note: There cannot be duplicate names (multiple ndarrays with the same name/identifier) in the final dataset. 
		///		  For merging multiple sources with the same name (e.g. inputs from different files) use merge extractors. 
		/// </summary>
		/// <param name="readData">The data read by any reader (requires the data formats to be compatible).</param>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <param name="handler">The computation handler to use for ndarray creation and manipulation.</param>
		/// <returns>The extracted named ndarrays, each containing a collection of numberOfRecords records (or less if unavailable). Return null when not a single record could be extracted.</returns>
		Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler);

		/// <summary>
		/// Extract a number of records from data read or extracted by any other source (requires the data formats to be compatible).
		/// Hierarchical extraction extracts from all parent extractors hierarchically and then extracts directly from that data.
		/// The ndarray format to use is BatchTimeFeatures (BTF), where T is 1 for non-sequential data. 
		/// Note: There cannot be duplicate names (multiple ndarrays with the same name/identifier) in the final dataset. 
		///		  For merging multiple sources with the same name (e.g. inputs from different files) use merge extractors. 
		/// </summary>
		/// <param name="readData">The data read by any reader (requires the data formats to be compatible).</param>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <param name="handler">The computation handler to use for ndarray creation and manipulation.</param>
		/// <returns>The extracted named ndarrays, each containing a collection of numberOfRecords records (or less if unavailable). Return null when not a single record could be extracted.</returns>
		Dictionary<string, INDArray> ExtractHierarchicalFrom(object readData, int numberOfRecords, IComputationHandler handler);

		/// <summary>
		/// Prepare this record extractor and its underlying resources for extraction.
		/// Note: This function may be called more than once (and subsequent calls should probably be ignored, depending on the implementation). 
		/// </summary>
		void Prepare();

		/// <summary>
		/// Attach a number of preprocessors to this extractor. Preprocessors are applied and their data is passed on in the order they were passed.
		/// </summary>
		/// <param name="preprocessors">The preprocessors to add to this extractor.</param>
		/// <returns>The last given preprocessor (for convenience).</returns>
		IRecordPreprocessor Preprocess(params IRecordPreprocessor[] preprocessors);

		/// <summary>
		/// Attach a number of extractors to this extractor. Extractors are applied and their data is passed on in the order they were passed.
		/// </summary>
		/// <param name="extractors">The extractors to add to this extractor.</param>
		/// <returns>The last given extractor (for convenience).</returns>
		IRecordExtractor Extractor(params IRecordExtractor[] extractors);
	}
}
