/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A registry resolver that resolves layered identifiers. Implementations are expected but not required to cache all resolved identifiers for better performance.
	/// </summary>
	public interface IRegistryResolver
	{
		/// <summary>
		/// The root of this resolver (where to start looking for identifiers).
		/// </summary>
		IRegistry Root { get; }

		/// <summary>
		/// Resolve all matching identifiers in this registry. 
		/// The supported notation syntax is:
		///		-	'.' separates registries hierarchically
		///			Example: "trainer2.training.accuracy"
		///		-	'*' indicates a wild-card mask, match any name - similar to regex's '.'
		///			Example: "trainer*.training.accuracy" match all sub-registries whose name starts with trainer
		///		-	'*<tag>' conditionally matching wild-card mask, match any name if the conditional tag
		///			Example: "*<trainer>.training.accuracy" match all sub-registries whose tags include the tag "trainer"
		/// </summary>
		/// <typeparam name="T">The most specific common type of the variables to retrieve.</typeparam>
		/// <param name="matchIdentifier">The full match identifier.</param>
		/// <param name="values">A reference to an array of values, filled with the values found at all matching identifiers.</param>
		/// <returns>A list of fully qualified matches to the match identifier.</returns>
		string[] ResolveRetrieve<T>(string matchIdentifier, ref T[] values);

		/// <summary>
		/// Set a single given value of a certain type to all matching identifiers. 
		/// Note: The individual registries might throw an exception if a type-protected value is set to the wrong type.
		/// </summary>
		/// <typeparam name="T">The type of the value.</typeparam>
		/// <param name="matchIdentifier">The full match identifier. </param>
		/// <param name="value"></param>
		/// <param name="associatedType">Optionally set the associated type (<see cref="IRegistry"/>)</param>
		/// <returns></returns>
		string[] ResolveSet<T>(string matchIdentifier, T value, System.Type associatedType = null);
	}
}
