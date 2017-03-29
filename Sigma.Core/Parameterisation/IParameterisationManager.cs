/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Parameterisation
{
	/// <summary>
	/// A parameterisation manager that maps parameter types by identifier, associated type (for registries) and actual type.
	/// Identifier takes precedence over associated type, associated type takes precedence over actual type.
	/// </summary>
	public interface IParameterisationManager
	{
		/// <summary>
		/// Add an identifier to parameter type mapping. Identifier mappings are strong.
		/// </summary>
		/// <param name="identifier">The identifier to map. No registry resolve notation, only direct identifiers.</param>
		/// <param name="type">The parameter type to map to.</param>
		void AddIdentifierMapping(string identifier, IParameterType type);

		/// <summary>
		/// Add an associated type to parameter type mapping. Associated type mappings are second strongest.
		/// </summary>
		/// <param name="associatedType">The associated type (as denoted in the registry).</param>
		/// <param name="type">The parameter type to map to.</param>
		void AddAssociatedTypeMapping(Type associatedType, IParameterType type);

		/// <summary>
		/// Add an actual type to parameter type mapping. Actual type mappings are weak.
		/// Note: If no matching type is found, the objects base types are traversed all the way up to <see cref="object"/>.
		/// </summary>
		/// <param name="actualType">The actual type.</param>
		/// <param name="type">The parameter type to map to.</param>
		void AddActualTypeMapping(Type actualType, IParameterType type);

		/// <summary>
		/// Get the parameter type for a certain registry entry.
		/// Note: This is just a convenience method which fetches the values and calls <see cref="GetParameterType(string,Type,Type)"/>
		/// </summary>
		/// <param name="identifier">The direct identifier within the given registry.</param>
		/// <param name="registry">The registry to fetch the value from.</param>
		/// <returns>The parameter type the fetched value from the given registry should be parameterised as.</returns>
		IParameterType GetParameterType(string identifier, IRegistry registry);

		/// <summary>
		/// Get the parameter type for a certain identifier, associated type (for registries) and actual type.
		/// The identifier itself cannot be null, the types can be null. 
		/// If no mapping is found, a KeyNotFoundException is thrown.
		/// </summary>
		/// <param name="identifier"></param>
		/// <param name="associatedType"></param>
		/// <param name="actualType"></param>
		/// <returns>The parameter type the fetched value from the given registry should be parameterised as.</returns>
		IParameterType GetParameterType(string identifier, Type associatedType, Type actualType);
	}
}
