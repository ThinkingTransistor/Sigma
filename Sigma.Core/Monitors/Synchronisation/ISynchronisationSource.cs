using System.Collections.Generic;

namespace Sigma.Core.Monitors.Synchronisation
{
	/// <summary>
	/// A synchronisation source provides additional data for a synchronisation handler if a value cannot be found in the default registry.
	/// </summary>
	public interface ISynchronisationSource
	{
		/// <summary>
		/// Try to retrieve a value from this source (if existent).
		/// </summary>
		/// <typeparam name="T">The type of the value that will be retrieved.</typeparam>
		/// <param name="key">The key of the value.</param>
		/// <param name="val">The value itself that will be assigned if it could be retrieved, <c>null</c> otherwise.</param>
		/// <returns><c>True</c> if the source could retrieve given key, <c>false</c> otherwise.</returns>
		bool TryGet<T>(string key, out T val);

		/// <summary>
		/// Try to set a value from this source (if existent).
		/// </summary>
		/// <typeparam name="T">The type of the value that will be set.</typeparam>
		/// <param name="key">The key of the value.</param>
		/// <param name="val">The value itself that will be assigned if it applicable.</param>
		/// <returns><c>True</c> if the source could set given key, <c>false</c> otherwise.</returns>
		bool TrySet<T>(string key, T val);

		/// <summary>
		/// Determine whether a given key is contained / manged by this source.
		/// </summary>
		/// <param name="key">The key that will be checked.</param>
		/// <returns><c>True</c> if given key can be accessed with get / set, <c>false</c> otherwise.</returns>
		bool Contains(string key);

		/// <summary>
		/// This is a list of keys this source provides. It is <b>completely</b> optional, although it is recommended to implement it.
		/// 
		/// Once a new source is added, the keys of the sources are checked against to determine double entries which makes debugging for users easier (as log entries are produced autoamtically).
		/// </summary>
		string[] Keys { get; }
	}
}