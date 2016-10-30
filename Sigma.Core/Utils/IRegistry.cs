using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A collection of keys and values with optional type checking.
	/// </summary>
	public interface IRegistry : ICollection, IEnumerable
	{
		/// <summary>
		/// Set a value with a given identifier. 
		/// </summary>
		/// <param name="identifier">The identifier.</param>
		/// <param name="value">The value to set.</param>
		/// <param name="valueType">Optionally set the type subsequent values with this identifier should have, unless explicitly overwritten.</param>
		void Set(String identifier, object value, Type valueType = null);

		/// <summary>
		/// Get the type-checked value associated with a given identifier.
		/// </summary>
		/// <typeparam name="T">The type the value should be cast to.</typeparam>
		/// <param name="identifier">The identifier.</param>
		/// <returns>The value of type T (if any) or null.</returns>
		T Get<T>(String identifier);

		/// <summary>
		/// Get the value associated with a given identifier.  
		/// </summary>
		/// <param name="identifier">The identifier.</param>
		/// <returns>The value (if any) or null.</returns>
		object Get(String identifier);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="matchIdentifier"></param>
		/// <param name="matchType"></param>
		/// <returns></returns>
		object[] GetAll(String matchIdentifier, Type matchType = null);

		/// <summary>
		/// Get all values.
		/// </summary>
		/// <returns>All values.</returns>
		object[] GetAllValues();

		/// <summary>
		/// Get all keys.
		/// </summary>
		/// <returns>All keys.</returns>
		string[] GetAllKeys();

		/// <summary>
		/// Quick hand operator overloaded syntax, <see cref="Get(String)" />
		/// </summary>
		/// <param name="identifier"></param>
		/// <returns></returns>
		object this[String identifier]
		{
			get; set;
		}
	}
}
