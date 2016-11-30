/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.MathAbstract
{
	/// <summary>
	/// A single mathematical value (i.e. number), used for interaction between ndarrays and handlers (is more expressive and faster). 
	/// </summary>
	public interface INumber 
	{
		/// <summary>
		/// The underlying value. 
		/// </summary>
		object Value { get; set; }
	}
}
