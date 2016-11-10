/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A cache provider which stores and loads named data of any format outside system memory. 
	/// </summary>
	public interface ICacheProvider : IDisposable
	{
		/// <summary>
		/// Store a serialisable object with a certain identifier. 
		/// </summary>
		/// <param name="identifier">The identifier.</param>
		/// <param name="data">The object to store (must be serialisable).</param>
		void Store(string identifier, object data);

		/// <summary>
		/// Load a cache object with certain identifier of a certain type.
		/// <typeparam name="T">The type the data should be cast to.</typeparam>
		/// </summary>
		/// <param name="identifier">The identifier.</param>
		/// <returns>The object with the given identifier if cached in this provider, otherwise null.</returns>
		T Load<T>(string identifier);

		/// <summary>
		/// Remove the cache entry associated with an identifier from this cache provider.
		/// </summary>
		/// <param name="identifier">The identifier for which the cache associated cache entry is to be removed.</param>
		void Remove(string identifier);

		/// <summary>
		/// Remove all cache entries associated with a cache provider.
		/// WARNING: Removing cache entries may cause certain datasets to load much more slowly or incorrectly. 
		///			 Use cases include removing cache entries for old datasets or datasets with different extractors. 
		/// </summary>
		void RemoveAll();

		/// <summary>
		/// Check whether an object with a certain identifier is cached in this provider. 
		/// </summary>
		/// <param name="identifier">The identifier to check for.</param>
		/// <returns>A boolean indicating whether the given identifier is cached in this provider.</returns>
		bool IsCached(string identifier);
	}
}
