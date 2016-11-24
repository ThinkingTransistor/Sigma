/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A neural network consisting of interconnected neural layers and a network architecture.
	/// </summary>
	public interface INetwork
	{
		/// <summary>
		/// The architecture of this network.
		/// </summary>
		INetworkArchitecture Architecture { get; set; }
	}
}
