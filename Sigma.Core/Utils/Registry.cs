/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A default implementation of the registry interface.
	/// A collection of keys and values (similar to a dictionary) where types and keys are registered for easier inspection. 
	/// Registries can be chained and represent a hierarchy, which can then be referred to using dot notation.
	/// </summary>
	public class Registry : IRegistry
	{
		internal Dictionary<string, object> MappedValues;
		internal Dictionary<string, Type> AssociatedTypes;

		public bool CheckTypes
		{
			get; set;
		} = true;

		public bool ExceptionOnCopyNonDeepCopyable
		{
			get; set;
		}

		public ICollection<string> Keys => MappedValues.Keys;

		public ICollection<object> Values => MappedValues.Values;

		public int Count => MappedValues.Count;

		public bool IsReadOnly => false;

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
			get; }

		public ISet<IRegistryHierarchyChangeListener> HierarchyChangeListeners
		{
			get; }

		/// <summary>
		/// Create a registry with a certain (optional) parent and an (optional) list of tags.
		/// </summary>
		/// <param name="parent">The optional parent to this registry.</param>
		/// <param name="tags">The optional tags to this registry.</param>
		public Registry(IRegistry parent = null, params string[] tags)
		{
			Parent = parent;
			Root = Parent?.Root == null ? Parent : Parent?.Root;

			MappedValues = new Dictionary<string, object>();
			AssociatedTypes = new Dictionary<string, Type>();

			if (tags == null)
			{
				tags = new string[0];
			}

			Tags = new HashSet<string>(tags);
			HierarchyChangeListeners = new HashSet<IRegistryHierarchyChangeListener>();
		}

		public object DeepCopy()
		{
			Registry copy = new Registry(Parent);

			foreach (string identifier in MappedValues.Keys)
			{
				object value = MappedValues[identifier];

				if (value.GetType().IsPrimitive)
				{
					copy.MappedValues.Add(identifier, value);
				}
				else
				{
					IDeepCopyable cloneable = value as IDeepCopyable;

					object copiedValue;

					if (cloneable == null)
					{
						if (!ExceptionOnCopyNonDeepCopyable)
						{
							copiedValue = value;

							if (value is ICloneable)
							{
								copiedValue = ((ICloneable) value).Clone();
							}
						}
						else
						{
							throw new InvalidOperationException($"The IRegistry.Copy method requires all non-primitive values to implement IDeepCopyable, but value {value} associated with identifier {identifier} does not.");
						}
					}
					else
					{
						copiedValue = cloneable.DeepCopy();
					}

					copy.MappedValues.Add(identifier, copiedValue);
				}
			}

			foreach (string identifier in AssociatedTypes.Keys)
			{
				copy.AssociatedTypes.Add(identifier, AssociatedTypes[identifier]);
			}

			return copy;
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

		public virtual void Set(string identifier, object value, Type valueType = null)
		{
			if (valueType != null)
			{
				if (!AssociatedTypes.ContainsKey(identifier))
				{
					AssociatedTypes.Add(identifier, valueType);
				}
				else
				{
					AssociatedTypes[identifier] = valueType;
				}
			}

			if (CheckTypes && AssociatedTypes.ContainsKey(identifier))
			{
				Type requiredType = AssociatedTypes[identifier];

				if (value.GetType() != requiredType && !value.GetType().IsSubclassOf(requiredType) && !requiredType.IsInstanceOfType(value))
				{
					throw new ArgumentException($"Values for identifier {identifier} must be of type {requiredType} (but given value {value} had type {value.GetType()})");
				}
			}

			//check if added object is another registry and if hierarchy listeners should be notified
			IRegistry valueAsRegistry = value as IRegistry;

			if (!MappedValues.ContainsKey(identifier))
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

					MappedValues[identifier] = value;

					if (previousValue != valueAsRegistry)
					{
						NotifyHierarchyChangeListeners(identifier, previousValue, valueAsRegistry);
					}
				}
				else
				{
					MappedValues[identifier] = value;
				}
			}
		}

		public virtual void Add(string key, object value)
		{
			MappedValues.Add(key, value);
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
			return (T) MappedValues[identifier];
		}

		public object Get(string identifier)
		{
			return MappedValues[identifier];
		}

		public T[] GetAllValues<T>(string matchIdentifier, Type matchType = null)
		{
			List<T> matchingValues = new List<T>();

			Regex regex = new Regex(matchIdentifier);

			foreach (string identifier in MappedValues.Keys)
			{
				if (regex.Match(identifier).Success && (matchType == null || matchType == MappedValues[identifier].GetType() || MappedValues[identifier].GetType().IsSubclassOf(matchType)))
				{
					matchingValues.Add((T) MappedValues[identifier]);
				}
			}

			return matchingValues.ToArray<T>();
		}

		public bool TryGetValue(string key, out object value)
		{
			return MappedValues.TryGetValue(key, out value);
		}

		public T Remove<T>(string identifier)
		{
			return (T) Remove(identifier);
		}

		public virtual object Remove(string identifier)
		{
			AssociatedTypes.Remove(identifier);

			object previousValue = MappedValues[identifier];

			MappedValues.Remove(identifier);

			return previousValue;
		}

		public virtual bool Remove(KeyValuePair<string, object> item)
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
			AssociatedTypes.Remove(key);

			return MappedValues.Remove(key);
		}

		public void Clear()
		{
			MappedValues.Clear();
			AssociatedTypes.Clear();
		}

		public bool ContainsKey(string key)
		{
			return MappedValues.ContainsKey(key);
		}

		public bool Contains(string key, object value)
		{
			return Contains(new KeyValuePair<string, object>(key, value));
		}

		public bool Contains(KeyValuePair<string, object> item)
		{
			return MappedValues.Contains(item);
		}

		public void CopyTo(KeyValuePair<string, object>[] array, int arrayIndex)
		{
			throw new NotImplementedException();
		}

		public IEnumerator<KeyValuePair<string, object>> GetEnumerator()
		{
			return MappedValues.GetEnumerator();
		}

		IEnumerator IEnumerable.GetEnumerator()
		{
			return MappedValues.GetEnumerator();
		}

		public IEnumerator GetKeyIterator()
		{
			return MappedValues.Keys.GetEnumerator();
		}

		public IEnumerator GetValueIterator()
		{
			return MappedValues.Values.GetEnumerator();
		}

		public override string ToString()
		{
			StringBuilder str = new StringBuilder();

			str.Append("\n[Registry]");
			str.Append("\n[Tags] = " + (Tags.Count == 0 ? "<none>" : (string.Join("", Tags))));

			foreach (var mappedValue in MappedValues)
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

	/// <summary>
	/// A read only view of a registry. 
	/// </summary>
	public class ReadOnlyRegistry : Registry, IReadOnlyRegistry
	{
		private Registry _underlyingRegistry;

		public ReadOnlyRegistry(Registry underlyingRegistry)
		{
			if (underlyingRegistry == null)
			{
				throw new ArgumentNullException("Underlying registry cannot be null.");
			}

			_underlyingRegistry = underlyingRegistry;
			MappedValues = underlyingRegistry.MappedValues;
			AssociatedTypes = underlyingRegistry.AssociatedTypes;
		}

		public override void Set(string identifier, object value, Type valueType = null)
		{
			throw new ReadOnlyException($"This registry is read-only. The identifier {identifier} cannot be set to {value}.");
		}

		public override void Add(string identifier, object value)
		{
			throw new ReadOnlyException($"This registry is read-only. The identifier {identifier} cannot be added.");
		}

		public override bool Remove(KeyValuePair<string, object> item)
		{
			throw new ReadOnlyException($"This registry is read-only. The item {item} cannot be removed.");
		}

		public override object Remove(string identifier)
		{
			throw new ReadOnlyException($"This registry is read-only. The identifier {identifier} cannot be removed.");
		}
	}
}
