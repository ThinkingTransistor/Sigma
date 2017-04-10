/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace Sigma.Core.Parameterisation
{
	/// <summary>
	/// A parameterisation 
	/// </summary>
	[Serializable]
	public class ParameterisationManager : IParameterisationManager
	{
		private readonly IDictionary<string, IParameterType> _identifierMappings;
		private readonly IDictionary<Type, IParameterType> _associatedTypeMappings;
		private readonly IDictionary<Type, IParameterType> _actualTypeMappings;

		/// <summary>
		/// Create a new parameterisation manager.
		/// </summary>
		public ParameterisationManager()
		{
			_identifierMappings = new ConcurrentDictionary<string, IParameterType>();
			_associatedTypeMappings = new ConcurrentDictionary<Type, IParameterType>();
			_actualTypeMappings = new ConcurrentDictionary<Type, IParameterType>();
		}

		/// <summary>
		/// Add an identifier to parameter type mapping. Identifier mappings are strong.
		/// </summary>
		/// <param name="identifier">The identifier to map. No registry resolve notation, only direct identifiers.</param>
		/// <param name="type">The parameter type to map to.</param>
		public void AddIdentifierMapping(string identifier, IParameterType type)
		{
			if (identifier == null) throw new ArgumentNullException(nameof(identifier));
			if (type == null) throw new ArgumentNullException(nameof(type));

			if (_identifierMappings.ContainsKey(identifier))
			{
				throw new InvalidOperationException($"Cannot add duplicate identifier to parameter type mapping for identifier {identifier}" +
				                                    $" and type {type}, identifier is already mapped to {_identifierMappings[identifier]}.");
			}

			_identifierMappings.Add(identifier, type);
		}

		/// <summary>
		/// Add an associated type to parameter type mapping. Associated type mappings are second strongest.
		/// </summary>
		/// <param name="associatedType">The associated type (as denoted in the registry).</param>
		/// <param name="type">The parameter type to map to.</param>
		public void AddAssociatedTypeMapping(Type associatedType, IParameterType type)
		{
			if (associatedType == null) throw new ArgumentNullException(nameof(associatedType));
			if (type == null) throw new ArgumentNullException(nameof(type));

			if (_associatedTypeMappings.ContainsKey(associatedType))
			{
				throw new InvalidOperationException($"Cannot add duplicate associated type to parameter type mapping for associated type {associatedType}" +
													$" and type {type}, associated type is already mapped to {_associatedTypeMappings[associatedType]}.");
			}

			_associatedTypeMappings.Add(associatedType, type);
		}

		/// <summary>
		/// Add an actual type to parameter type mapping. Actual type mappings are weak.
		/// Note: If no matching type is found, the objects base types are traversed all the way up to <see cref="object"/>.
		/// </summary>
		/// <param name="actualType">The actual type.</param>
		/// <param name="type">The parameter type to map to.</param>
		public void AddActualTypeMapping(Type actualType, IParameterType type)
		{
			if (actualType == null) throw new ArgumentNullException(nameof(actualType));
			if (type == null) throw new ArgumentNullException(nameof(type));

			if (_actualTypeMappings.ContainsKey(actualType))
			{
				throw new InvalidOperationException($"Cannot add duplicate actual type to parameter type mapping for associated type {actualType}" +
													$" and type {type}, actual type is already mapped to {_actualTypeMappings[actualType]}.");
			}

			_actualTypeMappings.Add(actualType, type);

		}

		/// <summary>
		/// Get the parameter type for a certain registry entry.
		/// Note: This is just a convenience method which fetches the values and calls <see cref="IParameterisationManager.GetParameterType(string,System.Type,System.Type)"/>
		/// </summary>
		/// <param name="identifier">The direct identifier within the given registry.</param>
		/// <param name="registry">The registry to fetch the value from.</param>
		/// <returns>The parameter type the fetched value from the given registry should be parameterised as.</returns>
		public IParameterType GetParameterType(string identifier, IRegistry registry)
		{
			if (identifier == null) throw new ArgumentNullException(nameof(identifier));
			if (registry == null) throw new ArgumentNullException(nameof(registry));

			if (!registry.ContainsKey(identifier))
			{
				throw new KeyNotFoundException($"Identifier {identifier} does not exist in given registry.");
			}

			return GetParameterType(identifier, registry.GetAssociatedType(identifier), registry[identifier].GetType());
		}

		/// <summary>
		/// Get the parameter type for a certain identifier, associated type (for registries) and actual type.
		/// The identifier itself cannot be null, the types can be null. 
		/// If no mapping is found, a KeyNotFoundException is thrown.
		/// </summary>
		/// <param name="identifier"></param>
		/// <param name="associatedType"></param>
		/// <param name="actualType"></param>
		/// <returns>The parameter type the fetched value from the given registry should be parameterised as.</returns>
		public IParameterType GetParameterType(string identifier, Type associatedType, Type actualType)
		{
			if (identifier == null) throw new ArgumentNullException(nameof(identifier));

			if (_identifierMappings.ContainsKey(identifier))
			{
				return _identifierMappings[identifier];
			}

			if (associatedType != null && _associatedTypeMappings.ContainsKey(associatedType))
			{
				return _associatedTypeMappings[associatedType];
			}

			if (actualType != null)
			{
				Type baseType = actualType;
				do
				{
					if (_actualTypeMappings.ContainsKey(baseType))
					{
						return _actualTypeMappings[baseType];
					}

					baseType = actualType.BaseType;
				} while (baseType != null);
			}

			throw new ParameterTypeMappingNotFoundException($"No parameter type mapping for identifier {identifier}, associated type {associatedType}," +
			                                                $" and actual type {actualType} (and its base types) was found.");
		}
	}

	/// <summary>
	/// An exception that occurs when a parameter type mapping is not found.
	/// </summary>
	public class ParameterTypeMappingNotFoundException : KeyNotFoundException
	{
		/// <inheritdoc cref="KeyNotFoundException(string)"/>
		public ParameterTypeMappingNotFoundException(string message) : base(message)
		{
		}
	}
}
