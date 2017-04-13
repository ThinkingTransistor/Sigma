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
	}
}