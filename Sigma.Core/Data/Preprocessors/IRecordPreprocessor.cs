/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Extractors;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// Preprocessors are a type of extractor which modify the actual data instead of just reshaping it. 
	/// They are in their own namespace separate from record extractors for a clearer conceptual distinction.
	/// </summary>
	public interface IRecordPreprocessor : IRecordExtractor
	{
		/// <summary>
		/// A flag indicating whether this preprocessor only affects the underlying data or also the shape of the data.
		/// Note: Most preprocessors only change the data (e.g. for normalisation) and not the data shape. 
		///		  This flag is useful for ordering, optimising and analysing preprocessing operations. 
		/// </summary>
		bool AffectsDataShape { get; }
	}
}
