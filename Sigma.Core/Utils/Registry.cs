using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Utils
{
	public class Registry : IRegistry
	{
		private Dictionary<string, object> mappedValues;
		private Dictionary<string, Type> associatedTypes;

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
			this.Root = Parent?.Parent;
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
				associatedTypes.Add(identifier, valueType);
			}

			mappedValues.Add(identifier, value);
		}

		public void Add(string key, object value)
		{
			Set(key, value);
		}

		public void Add(KeyValuePair<string, object> item)
		{
			Set(item.Key, item.Value);
		}

		public T Get<T>(string identifier)
		{
			return (T) mappedValues[identifier];
		}

		public object Get(string identifier)
		{
			return mappedValues[identifier];
		}

		public object[] GetAllValues(string matchIdentifier, Type matchType = null)
		{
			throw new NotImplementedException();
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

			return mappedValues.Remove(identifier);
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
