/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

#pragma warning disable 1570
namespace Sigma.Core.Utils
{
	/// <summary>
	/// A registry resolver that resolves layered identifiers. Implementations are expected but not required to cache all resolved identifiers for better performance.
	/// The supported notation syntax is:
	///		-	'.' separates registries hierarchically
	///			Example: "trainer2.training.accuracy"
	///		-	'*' indicates a wild-card mask, match any name - similar to regex's '.'
	///			Example: "trainer*.training.accuracy" match all sub-registries whose name starts with trainer
	///		-	'*<tag>' conditionally matching wild-card mask, match any name if the conditional tag
	///			Example: "*<trainer>.training.accuracy" match all sub-registries whose tags include the tag "trainer"
	///</summary>
	public interface IRegistryResolver
	{
		/// <summary>
		/// The root of this resolver (where to start looking for identifiers).
		/// </summary>
		IRegistry Root { get; }

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="fullMatchedIdentifierArray">The fully matched identifiers corresponding to the given match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		T[] ResolveGet<T>(string matchIdentifier, out string[] fullMatchedIdentifierArray, T[] values = null);

		/// <summary>
		/// Resolve all matching identifiers in this registry. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="values">An array of values found at the matching identifiers, filled with the values found at all matching identifiers (for reuse and optimisation if request is issued repeatedly).</param>
		/// <returns>An array of values found at the matching identifiers. The parameter values is used if it is large enough and not null.</returns>
		T[] ResolveGet<T>(string matchIdentifier, T[] values = null);

		/// <summary>
		/// Set a single given value of a certain type to all matching identifiers. For the detailed supported syntax <see cref="IRegistryResolver"/>.
		/// Note: The individual registries might throw an exception if a type-protected value is set to the wrong type.
		/// </summary>
		/// <typeparam name="T">The type of the value.</typeparam>
		/// <param name="matchIdentifier">The full match identifier. </param>
		/// <param name="value"></param>
		/// <param name="associatedType">Optionally set the associated type (<see cref="IRegistry"/>)</param>
		/// <returns>A list of fully qualified matches to the match identifier.</returns>
		string[] ResolveSet<T>(string matchIdentifier, T value, System.Type associatedType = null);
	}
}
