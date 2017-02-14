/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;

namespace Sigma.Core.MathAbstract
{
	/// <summary>
	/// A single traceable mathematical value (i.e. number), used for interaction between ndarrays and handlers (is more expressive and faster). 
	/// </summary>
	public interface INumber : ITraceable, IDeepCopyable
	{
		/// <summary>
		/// The underlying value. 
		/// </summary>
		object Value { get; set; }

		/// <summary>
		/// Get the underlying value as another type.
		/// </summary>
		/// <typeparam name="TOther">The other type.</typeparam>
		/// <returns>The value as the other type.</returns>
		TOther GetValueAs<TOther>();
	}
}
