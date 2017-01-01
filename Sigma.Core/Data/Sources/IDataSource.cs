/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.IO;

namespace Sigma.Core.Data.Sources
{
	/// <summary>
	/// A data source (e.g. local file source or URL web source) which can be used to retrieve data streams (e.g. by record readers to populate datasets).
	/// </summary>
	public interface IDataSource : IDisposable
	{
		/// <summary>
		/// The name of this source (e.g. file name, URL).
		/// </summary>
		string ResourceName { get; }

		/// <summary>
		/// Indicates whether this is a seekable data source.
		/// Seekable sources can be loaded and reloaded partially, non-chunkable sources are pre-fetched in <see cref="Prepare"/>.
		/// </summary>
		bool Seekable { get; }

		/// <summary>
		/// Check whether the specified data source exists and can be retrieved.
		/// </summary>
		/// <returns>A boolean indicating whether the specified data set source exists.</returns>
		bool Exists();

		/// <summary>
		/// Prepares this data source for retrieval via <see cref="Retrieve"/>. 
		/// For non-chunkable sources this includes loading the entire data set.
		/// Note: This function may be called more than once (and subsequent calls should probably be ignored, depending on the implementation). 
		/// </summary>
		void Prepare();

		/// <summary>
		/// Retrieves the data source via a stream.
		/// </summary>
		/// <returns></returns>
		Stream Retrieve();
	}
}
