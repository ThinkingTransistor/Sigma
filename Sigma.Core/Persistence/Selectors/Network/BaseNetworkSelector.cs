/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using Sigma.Core.Architecture;
using Sigma.Core.Utils;

namespace Sigma.Core.Persistence.Selectors.Network
{
	/// <summary>
	/// A base network selector that handles basic <see cref="NetworkComponent"/> selection independent of actual network type.
	/// </summary>
	/// <typeparam name="TNetwork"></typeparam>
	public abstract class BaseNetworkSelector<TNetwork> : INetworkSelector<TNetwork> where TNetwork : INetwork
	{
		/// <summary>
		/// The current result object of type <see cref="INetwork"/>.
		/// </summary>
		public TNetwork Result { get; }

		/// <summary>
		/// Create a base network selector for a network.
		/// </summary>
		/// <param name="result">The network.</param>
		protected BaseNetworkSelector(TNetwork result)
		{
			if (result == null) throw new ArgumentNullException(nameof(result));

			Result = result;
		}

		/// <summary>
		/// Get the "emptiest" available version of this <see cref="TNetwork"/> while still retaining a legal object state for type <see cref="TNetwork"/>.
		/// Note: Any parameters that are fixed per-object (such as unique name) must be retained.
		/// </summary>
		/// <returns>An empty version of the object.</returns>
		public ISelector<TNetwork> Empty()
		{
			return Keep(NetworkComponent.None);
		}

		/// <summary>
		/// Get an uninitialised version of this <see cref="TNetwork"/> without any runtime information, but ready to be re-initialised.
		/// </summary>
		/// <returns></returns>
		public ISelector<TNetwork> Uninitialised()
		{
			return Keep(NetworkComponent.Architecture);
		}

		/// <summary>
		/// Keep a set of network components, discard all other components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new network with the given component(s) retained.</returns>
		public ISelector<TNetwork> Keep(params NetworkComponent[] components)
		{
			if (components.ContainsFlag(NetworkComponent.Everything))
			{
				return CreateSelector((TNetwork) Result.DeepCopy());
			}

			TNetwork network = CreateNetwork(Result.Name);

			if (components.ContainsFlag(NetworkComponent.Architecture))
			{				
				network.Architecture = (INetworkArchitecture) Result.Architecture.DeepCopy();
			}

			if (components.ContainsFlag(NetworkComponent.Parameters))
			{
				Result.TransferParametersTo(network);
			}

			return CreateSelector(network);
		}

		/// <summary>
		/// Discard specified network components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to discard.</param>
		/// <returns>A selector for a new network with the given component(s) discarded.</returns>
		public ISelector<TNetwork> Discard(params NetworkComponent[] components)
		{
			if (components.ContainsFlag(NetworkComponent.Everything))
			{
				return Keep(NetworkComponent.None);
			}

			if (components.ContainsFlag(NetworkComponent.Architecture))
			{
				throw new InvalidOperationException($"Cannot only discard architecture and keep everything, that does not make sense.");
			}

			throw new InvalidOperationException($"Cannot discard given components {components}, discard is invalid and probably does not make sense.");
		}

		/// <summary>
		/// Create a network of this network selectors network type with a certain name.
		/// </summary>
		/// <param name="name">The name.</param>
		/// <returns>A network of the appropriate time with the given name.</returns>
		protected abstract TNetwork CreateNetwork(string name);

	    /// <summary>
	    /// Create a network selector with a certain network.
	    /// </summary>
	    /// <param name="network">The network.</param>
	    /// <returns>A network of the appropriate time with the given name.</returns>
	    protected abstract INetworkSelector<TNetwork> CreateSelector(TNetwork network);
	}
}
