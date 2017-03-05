/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

namespace Sigma.Core.Persistence.Selectors.Network
{
	public class DefaultNetworkSelector : BaseNetworkSelector<Architecture.Network>
	{
		/// <summary>
		/// Create a base network selector for a network.
		/// </summary>
		/// <param name="result">The network.</param>
		public DefaultNetworkSelector(Architecture.Network result) : base(result)
		{
		}

		protected override Architecture.Network CreateNetwork(string name)
		{
			return new Architecture.Network(name);
		}

		protected override INetworkSelector<Architecture.Network> CreateSelector(Architecture.Network network)
		{
			return new DefaultNetworkSelector(network);
		}
	}
}
