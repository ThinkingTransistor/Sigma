/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
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

		public IRegistry Parent
		{
			get; set;
		}

		public IRegistry Root
		{
			get; set;
		}

		public ISet<string> Tags
		{
			get; private set;
		}

		public ISet<IRegistryHierarchyChangeListener> HierarchyChangeListeners
		{
			get; private set;
		}

		public Registry(IRegistry parent = null, params string[] tags)
		{
			this.Parent = parent;
			this.Root = Parent?.Root == null ? Parent : Parent?.Root;

			this.mappedValues = new Dictionary<string, object>();
			this.associatedTypes = new Dictionary<string, Type>();

			if (tags == null)
			{
				tags = new string[0];
			}

			this.Tags = new HashSet<string>(tags);
			this.HierarchyChangeListeners = new HashSet<IRegistryHierarchyChangeListener>();
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

			if (CheckTypes && associatedTypes.ContainsKey(identifier))
			{
				Type requiredType = associatedTypes[identifier];

				if (value.GetType() != requiredType && !value.GetType().IsSubclassOf(requiredType) && !requiredType.IsAssignableFrom(value.GetType()))
				{
					throw new ArgumentException($"Values for identifier {identifier} must be of type {requiredType} (but given value {value} had type {value?.GetType()})");
				}
			}

			//check if added object is another registry and if hierarchy listeners should be notified
			IRegistry valueAsRegistry = value as IRegistry;

			if (!mappedValues.ContainsKey(identifier))
			{
				Add(identifier, value);

				//notify if value is of type IRegistry
				if (valueAsRegistry != null)
				{
					NotifyHierarchyChangeListeners(identifier, null, valueAsRegistry);
				}
			}
			else
			{
				//notify if value is of type IRegistry and if value changed
				if (valueAsRegistry != null)
				{
					IRegistry previousValue = this[identifier] as IRegistry;

					mappedValues[identifier] = value;

					if (previousValue != valueAsRegistry)
					{
						NotifyHierarchyChangeListeners(identifier, previousValue, valueAsRegistry);
					}
				}
				else
				{
					mappedValues[identifier] = value;
				}
			}
		}

		public void Add(string key, object value)
		{
			mappedValues.Add(key, value);
		}

		public void Add(KeyValuePair<string, object> item)
		{
			Add(item.Key, item.Value);
		}

		private void NotifyHierarchyChangeListeners(string identifier, IRegistry previousChild, IRegistry newChild)
		{
			foreach (IRegistryHierarchyChangeListener listener in HierarchyChangeListeners)
			{
				listener.OnChildHierarchyChanged(identifier, previousChild, newChild);
			}
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
			if (ReferenceEquals(Get(item.Key), item.Value))
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

		public bool Contains(string key, object value)
		{
			return Contains(new KeyValuePair<string, object>(key, value));
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

		public override string ToString()
		{
			StringBuilder str = new StringBuilder();

			str.Append("\n[Registry]");
			str.Append("\n[Tags] = " + (Tags.Count == 0 ? "<none>" : (string.Join("", Tags))));

			foreach (var mappedValue in mappedValues)
			{
				if (mappedValue.Value is IRegistry)
				{
					str.Append(mappedValue.Value.ToString().Replace("\n", "\n\t"));
				}
				else
				{
					str.Append($"\n[{mappedValue.Key}] = {mappedValue.Value}");
				}
			}

			return str.ToString();
		}
	}
}
