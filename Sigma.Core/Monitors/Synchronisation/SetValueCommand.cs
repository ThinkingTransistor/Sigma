using System;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.Synchronisation
{
	/// <summary>
	/// A default implementation of a command that sets registry keys to generic values. 
	/// </summary>
	[Serializable]
	public class SetValueCommand<T> : BaseCommand
	{
		private const string KeyIdentifier = "keys";
		private const string ValueIdentifier = "values";
		private const string AddItentifierIfNotExistsIdentifier = "identifier";

		/// <summary>
		/// Determines whether identifiers are added when they do not exist.
		/// </summary>
		public bool AddItentifierIfNotExists
		{
			get { return (bool) ParameterRegistry[AddItentifierIfNotExistsIdentifier]; }
			set { ParameterRegistry[AddItentifierIfNotExistsIdentifier] = value; }
		}

		/// <summary>
		/// Create a command that sets one registry key to a value.
		/// </summary>
		/// <param name="key">The registry key. It has to be fully resolved.</param>
		/// <param name="value">The new value the registry[key] will get.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string key, T value, Action onFinish = null) : this(new[] { key }, new[] { value }, onFinish) { }

		/// <summary>
		/// Create a command that sets two registry keys with two individual values.
		/// </summary>
		/// <param name="key1">The first registry key. It has to be fully resolved.</param>
		/// <param name="value1">The first new value the registry[key1] will get.</param>
		/// <param name="key2">The second registry key. It has to be fully resolved.</param>
		/// <param name="value2">The second new value the registry[key2] will get.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string key1, T value1, string key2, T value2, Action onFinish = null) : this(new[] { key1, key2 }, new[] { value1, value2 }, onFinish) { }

		/// <summary>
		/// Create a command that sets multiple registry keys to the same value.
		/// </summary>
		/// <param name="keys">The registry keys that will be modified. They have to be fully resolved.</param>
		/// <param name="value">The value each registry[key] will be set to.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string[] keys, T value, Action onFinish = null) : this(keys, new[] { value }, onFinish) { }

		/// <summary>
		/// Create a command that sets multiple registry kess to multiple different values. 
		/// Both arrays may not be <c>null</c>, nor empty, nor differ in length.
		/// </summary>
		/// <param name="keys">The registry keys that will be modified. They have to be fully resolved.</param>
		/// <param name="values">The values that will be set to the registry. The first key will receive the first value and so on.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string[] keys, T[] values, Action onFinish = null) : base(onFinish, keys)
		{
			if (keys == null) throw new ArgumentNullException(nameof(keys));
			if (values == null) throw new ArgumentNullException(nameof(values));

			if (keys.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(keys));
			if (values.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(values));


			// if there is only one value, all keys will receive this value
			// if the lengths are different, we don't know what to do
			if (values.Length != 1 && keys.Length != values.Length) throw new ArgumentException($"{nameof(keys)} and {nameof(values)} have different lengths.", nameof(keys));

			// set default values
			AddItentifierIfNotExists = false;

			// expand the values 
			if (values.Length == 1)
			{
				T[] expandedValues = new T[keys.Length];
				for (int i = 0; i < expandedValues.Length; i++)
				{
					expandedValues[i] = values[0];
				}

				values = expandedValues;
			}

			//store keys and values in the parameter registry
			ParameterRegistry[KeyIdentifier] = keys;
			ParameterRegistry[ValueIdentifier] = values;
		}


		/// <summary>
		/// Invoke this command and set all required values. 
		/// </summary>
		/// <param name="registry">The registry containing the required values for this command's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			string[] keys = (string[]) ParameterRegistry[KeyIdentifier];
			T[] values = (T[]) ParameterRegistry[ValueIdentifier];

			for (int i = 0; i < keys.Length; i++)
			{
				resolver.ResolveSet(keys[i], values[i], AddItentifierIfNotExists, typeof(T));
			}
		}
	}

	#region NonGeneric

	/// <summary>
	/// A default implementation of a command that sets registry keys to non generic values. 
	/// </summary>
	[Serializable]
	public class SetValueCommand : SetValueCommand<object>
	{
		/// <summary>
		/// Create a command that sets one registry key to a value.
		/// </summary>
		/// <param name="key">The registry key. It has to be fully resolved.</param>
		/// <param name="value">The new value the registry[key] will get.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string key, object value, Action onFinish = null) : base(key, value, onFinish)
		{
		}

		/// <summary>
		/// Create a command that sets two registry keys with two individual values.
		/// </summary>
		/// <param name="key1">The first registry key. It has to be fully resolved.</param>
		/// <param name="value1">The first new value the registry[key1] will get.</param>
		/// <param name="key2">The second registry key. It has to be fully resolved.</param>
		/// <param name="value2">The second new value the registry[key2] will get.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string key1, object value1, string key2, object value2, Action onFinish = null) : base(key1, value1, key2, value2, onFinish)
		{
		}

		/// <summary>
		/// Create a command that sets multiple registry keys to the same value.
		/// </summary>
		/// <param name="keys">The registry keys that will be modified. They have to be fully resolved.</param>
		/// <param name="value">The value each registry[key] will be set to.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string[] keys, object value, Action onFinish = null) : base(keys, value, onFinish)
		{
		}

		/// <summary>
		/// Create a command that sets multiple registry kess to multiple different values. 
		/// Both arrays may not be <c>null</c>, nor empty, nor differ in length.
		/// </summary>
		/// <param name="keys">The registry keys that will be modified. They have to be fully resolved.</param>
		/// <param name="values">The values that will be set to the registry. The first key will receive the first value and so on.</param>
		/// <param name="onFinish">The function that will be called when the execution has been finished. If <c>null</c>, no function will be called.</param>
		public SetValueCommand(string[] keys, object[] values, Action onFinish = null) : base(keys, values, onFinish)
		{
		}
	}

	#endregion
}