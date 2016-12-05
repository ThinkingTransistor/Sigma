/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;

namespace Sigma.Core.MathAbstract
{
	/// <summary>
	/// A traceable mathematical object.
	/// </summary>
	public interface ITraceable
	{
		/// <summary>
		/// The computation handler associated with this traceable object (null if none).
		/// </summary>
		IComputationHandler AssociatedHandler { get; }
	}
}
