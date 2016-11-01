﻿using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Sigma.Core.Utils
{
	public class Registry : IRegistry
	{
		private Dictionary<string, object> mappedValues;
		private Dictionary<string, Type> associatedTypes;

		public bool CheckTypes
		{
			get; set;
		} = true;

		public Registry Parent
		{
			get; private set;
		}

		public Registry Root
		{
			get; private set;
		}

		public ICollection<string> Keys
		{
			get
			{
				return mappedValues.Keys;
			}
		}

		public ICollection<object> Values
		{
			get
			{
				return mappedValues.Values;
			}
		}

		public int Count
		{
			get
			{
				return mappedValues.Count;
			}
		}

		public bool IsReadOnly
		{
			get
			{
				return false;
			}
		}

		public Registry(Registry parent = null)
		{
			this.mappedValues = new Dictionary<string, object>();
			this.associatedTypes = new Dictionary<string, Type>();
			this.Parent = parent;
			this.Root = Parent?.Root == null ? Parent : Parent?.Root;
		}

		public object this[string identifier]
		{
			get
			{
				return Get(identifier);
			}

			set
			{
				Set(identifier, value);
			}
		}

		public void Set(string identifier, object value, Type valueType = null)
		{
			if (valueType != null)
			{
				if (!associatedTypes.ContainsKey(identifier))
				{
					associatedTypes.Add(identifier, valueType);
				}
				else
				{
					associatedTypes[identifier] = valueType;
				}
			}

			if (CheckTypes && associatedTypes.ContainsKey(identifier) && (value.GetType() != associatedTypes[identifier] && !value.GetType().IsSubclassOf(associatedTypes[identifier])))
			{
				throw new ArgumentException($"Values for identifier {identifier} must be of type {associatedTypes[identifier]} (but given value {value} had type {value?.GetType()})");
			}

			if (!mappedValues.ContainsKey(identifier))
			{
				mappedValues.Add(identifier, value);
			}
			else
			{
				mappedValues[identifier] = value;
			}
		}

		public void Add(string key, object value)
		{
			mappedValues.Add(key, value);
		}

		public void Add(KeyValuePair<string, object> item)
		{
			mappedValues.Add(item.Key, item.Value);
		}

		public T Get<T>(string identifier)
		{
			return (T) mappedValues[identifier];
		}

		public object Get(string identifier)
		{
			return mappedValues[identifier];
		}

		public T[] GetAllValues<T>(string matchIdentifier, Type matchType = null)
		{
			List<T> matchingValues = new List<T>();

			Regex regex = new Regex(matchIdentifier);

			foreach (string identifier in mappedValues.Keys)
			{
				if (regex.Match(identifier).Success && (matchType == null || matchType == mappedValues[identifier].GetType() || mappedValues[identifier].GetType().IsSubclassOf(matchType)))
				{
					matchingValues.Add((T) mappedValues[identifier]);
				}
			}

			return matchingValues.ToArray<T>();
		}

		public bool TryGetValue(string key, out object value)
		{
			return mappedValues.TryGetValue(key, out value);
		}

		public T Remove<T>(string identifier)
		{
			return (T) Remove(identifier);
		}

		public object Remove(string identifier)
		{
			associatedTypes.Remove(identifier);

			object previousValue = mappedValues[identifier];

			mappedValues.Remove(identifier);

			return previousValue;
		}

		public bool Remove(KeyValuePair<string, object> item)
		{
			if (Object.ReferenceEquals(Get(item.Key), item.Value))
			{
				Remove(item.Key);

				return true;
			}

			return false;
		}

		bool IDictionary<string, object>.Remove(string key)
		{
			associatedTypes.Remove(key);

			return mappedValues.Remove(key);
		}

		public void Clear()
		{
			mappedValues.Clear();
			associatedTypes.Clear();
		}

		public bool ContainsKey(string key)
		{
			return mappedValues.ContainsKey(key);
		}

		public bool Contains(KeyValuePair<string, object> item)
		{
			return mappedValues.Contains(item);
		}

		public void CopyTo(KeyValuePair<string, object>[] array, int arrayIndex)
		{
			throw new NotImplementedException();
		}

		public IEnumerator<KeyValuePair<string, object>> GetEnumerator()
		{
			return mappedValues.GetEnumerator();
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return mappedValues.GetEnumerator();
		}

		public IEnumerator GetKeyIterator()
		{
			return mappedValues.Keys.GetEnumerator();
		}

		public IEnumerator GetValueIterator()
		{
			return mappedValues.Values.GetEnumerator();
		}
	}
}