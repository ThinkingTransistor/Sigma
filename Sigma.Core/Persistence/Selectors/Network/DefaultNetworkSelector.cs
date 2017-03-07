/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Architecture;

namespace Sigma.Core.Persistence.Selectors.Network
{
	/// <summary>
	/// A default network selector, assuming the availability of a Network(string name) constructor.
	/// </summary>
	public class DefaultNetworkSelector<TNetwork> : BaseNetworkSelector<TNetwork> where TNetwork : INetwork
	{
		/// <summary>
		/// Create a base network selector for a network.
		/// Note: The network type must have a Network(string name) constructor.
		/// </summary>
		/// <param name="result">The network.</param>
		public DefaultNetworkSelector(TNetwork result) : base(result)
		{
		}

		/// <inheritdoc cref="BaseNetworkSelector{TNetwork}.CreateNetwork"/>
		protected override TNetwork CreateNetwork(string name)
		{
			return (TNetwork) Activator.CreateInstance(Result.GetType(), name);
		}

		/// <inheritdoc cref="BaseNetworkSelector{TNetwork}.CreateSelector"/>
		protected override INetworkSelector<TNetwork> CreateSelector(TNetwork network)
		{
			return new DefaultNetworkSelector<TNetwork>(network);
		}
	}
}
