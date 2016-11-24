/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A network architecture from which an actual network can be constructed.
	/// </summary>
	public interface INetworkArchitecture
	{
		/// <summary>
		/// The total number of layers in this architecture.
		/// </summary>
		int LayerCount { get; }

		/// <summary>
		/// Yield all layers in the order they should be processed. 
		/// </summary>
		/// <returns></returns>
		IEnumerable<LayerConstruct> YieldLayerConstructs();
	}
}
