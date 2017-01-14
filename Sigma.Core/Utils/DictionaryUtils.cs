using System;
using System.Collections.Generic;

namespace Sigma.Core.Utils
{
	public static class DictionaryUtils
	{
		/// <summary>
		/// This method is equivalent to <see cref="IDictionary{TKey,TValue}.TryGetValue"/> but it adds the key
		/// to the dictionary with the <see cref="createFunction"/> if not existent.
		/// </summary>
		/// <typeparam name="K">The type of they.</typeparam>
		/// <typeparam name="V">The type of the value-</typeparam>
		/// <param name="dictionary">The dictionary this extension method is for. </param>
		/// <param name="key">The key whose value to get.</param>
		/// <param name="value">When this method returns, the value associated with the key is returned. If the key is not present, the value
		/// will be created with <see cref="createFunction"/> and added to the dictionary.</param>
		/// <param name="createFunction">The <see cref="Func{TResult}"/> that will be used to generate the value if not present.</param>
		/// <returns><c>True</c> if the key was present before, <c>false</c> otherwise. </returns>
		public static bool TryGetValue<K, V>(this IDictionary<K, V> dictionary, K key, out V value, Func<V> createFunction)
		{
			if (createFunction == null)
			{
				throw new ArgumentNullException(nameof(createFunction));
			}

			if (!dictionary.TryGetValue(key, out value))
			{
				value = createFunction.Invoke();
				dictionary.Add(key, value);
				return false;
			}

			return true;
		}

		/// <summary>
		/// This method is equivalent to <see cref="IDictionary{TKey,TValue}.TryGetValue"/> but it adds the key
		/// to the dictionary with the <see cref="createFunction"/> if not existent (and returns instead of out). 
		/// </summary>
		/// <typeparam name="K">The type of they.</typeparam>
		/// <typeparam name="V">The type of the value-</typeparam>
		/// <param name="dictionary">The dictionary this extension method is for. </param>
		/// <param name="key">The key whose value to get.</param>
		/// <param name="createFunction">The <see cref="Func{TResult}"/> that will be used to generate the value if not present.</param>
		/// <returns>The value associated with the key. If they key is not present, the value will be created with <see cref="createFunction"/>, added to the dictionary and
		/// returned. </returns>
		public static V TryGetValue<K, V>(this IDictionary<K, V> dictionary, K key, Func<V> createFunction)
		{
			if (createFunction == null)
			{
				throw new ArgumentNullException(nameof(createFunction));
			}

			V value;

			if (!dictionary.TryGetValue(key, out value))
			{
				value = createFunction.Invoke();
				dictionary.Add(key, value);
			}

			return value;
		}
	}
}