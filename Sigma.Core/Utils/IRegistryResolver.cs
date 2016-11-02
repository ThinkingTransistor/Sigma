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
	interface IRegistryResolver
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
		///			Example: "*<Trainer>.training.accuracy" match all sub-registries whose 
		/// </summary>
		/// <typeparam name="T"></typeparam>
		/// <param name="matchIdentifier"></param>
		/// <param name="fullIdentifiers"></param>
		/// <returns></returns>
		string[] Resolve<T>(string matchIdentifier, ref T[] values);
	}
}
