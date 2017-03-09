/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Persistence.Selectors;

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

		/// <summary>
		/// Remove a value from a collection within a dictionary and remove the collection from the dictionary if it's empty.
		/// </summary>
		/// <typeparam name="K">The key type.</typeparam>
		/// <typeparam name="V">The value type.</typeparam>
		/// <param name="dictionary">The dictionary.</param>
		/// <param name="key">The key.</param>
		/// <param name="valueToRemove">The value to remove from the collection within the dictionary.</param>
		/// <returns>A boolean indicating if the collection within the dictionary was removed ("cleaned").</returns>
		public static bool RemoveAndClean<K, V>(this IDictionary<K, ISet<V>> dictionary, K key, V valueToRemove)
		{
			ICollection<V> collection = dictionary[key];

			if (collection == null)
			{
				throw new InvalidOperationException($"Cannot remove and clean collection in dictionary for non-existing key {key}.");
			}

			collection.Remove(valueToRemove);

			return collection.Count == 0 && dictionary.Remove(key);
		}

		/// <summary>
		/// Add all key value pairs from another dictionary to this dictionary.
		/// </summary>
		/// <typeparam name="K">The key type.</typeparam>
		/// <typeparam name="V">The value type.</typeparam>
		/// <param name="dictionary">This dictionary (where the entries will be added to).</param>
		/// <param name="other">The other dictionary (where the entries will be taken from).</param>
		public static void AddAll<K, V>(this IDictionary<K, V> dictionary, IDictionary<K, V> other)
		{
			if (other == null) throw new ArgumentNullException(nameof(other));

			foreach (var keypair in other)
			{
				dictionary.Add(keypair);
			}
		}

		/// <summary>
		/// Check if any component in an array of selector components contains a component id flag.
		/// </summary>
		/// <typeparam name="TComponent">The component type.</typeparam>
		/// <param name="enumeration">The enumeration.</param>
		/// <param name="component">The component (flag) to check for.</param>
		/// <returns>A boolean indicating whether the enumeration contains a component with the given components id flag.</returns>
		public static bool ContainsFlag<TComponent>(this IEnumerable<TComponent> enumeration, TComponent component) where TComponent : SelectorComponent
		{
			return enumeration.Any(element => (element.Id & component.Id) == component.Id);
		}
	}
}