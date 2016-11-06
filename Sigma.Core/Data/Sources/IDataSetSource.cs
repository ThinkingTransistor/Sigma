/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.IO;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A data set source (e.g. local file source or URL web source) which can be used by record readers to populate datasets.
	/// </summary>
	public interface IDataSetSource
	{
		/// <summary>
		/// Indicates whether this is a chunkable data set source.
		/// Chunkable sources can be loaded partially, non-chunkable sources are pre-fetched in <see cref="Prepare"/>.
		/// </summary>
		bool Chunkable { get; }

		/// <summary>
		/// Check whether the specified data set source exists and can be retrieved.
		/// </summary>
		/// <returns>A boolean indicating whether the specified data set source exists.</returns>
		bool Exists();

		/// <summary>
		/// Prepares this data set for retrieval via <see cref="Retrieve"/>. 
		/// For non-chunkable sources this includes loading the entire data set.
		/// </summary>
		void Prepare();

		/// <summary>
		/// Retrieves the data set source.
		/// </summary>
		/// <returns></returns>
		Stream Retrieve();
	}
}
