/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Extractors
{
	/// <summary>
	/// A record extractor which extracts records from a record reader into an INDArray.
	/// Example: CSVRecordReader reads IDataSetSource as CSV data, CSVRecordExtractor extracts specific entries into an INDArray.
	/// </summary>
	public interface IRecordExtractor
	{
		/// <summary>
		/// The record reader that is attached to this extractor.
		/// </summary>
		IRecordReader Reader { get; }

		/// <summary>
		/// Extract a number of records into an ndarray. 
		/// </summary>
		/// <param name="numberOfRecords">The number of records to extract.</param>
		/// <returns>The extracted ndarray.</returns>
		INDArray Extract(int numberOfRecords);

		/// <summary>
		/// Attach a number of preprocessors to this extractor. Preprocessors are applied in the order they were passed.
		/// </summary>
		/// <param name="preprocessors">The preprocessors to add to this extractor.</param>
		/// <returns>The last given preprocessor (for convenience).</returns>
		IRecordPreprocessor Preprocess(params IRecordPreprocessor[] preprocessors);
	}
}
