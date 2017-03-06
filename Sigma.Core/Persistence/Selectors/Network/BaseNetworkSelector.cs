/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using Sigma.Core.Architecture;

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
			if (components.Length != 1)
			{
				throw new ArgumentException($"Number of network components to keep must be 1 but was {components.Length} (either none, architecture or everything - they are mutually exclusive).");
			}

			if (components[0] == NetworkComponent.Everything)
			{
				return CreateSelector((TNetwork) Result.DeepCopy());
			}

			TNetwork network = CreateNetwork(Result.Name);

			if (components[0] == NetworkComponent.Architecture)
			{				
				network.Architecture = Result.Architecture;
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
			if (components.Contains(NetworkComponent.Everything))
			{
				return Keep(NetworkComponent.None);
			}

			if (components.Contains(NetworkComponent.Architecture))
			{
				throw new InvalidOperationException($"Cannot only discard architecture and keep everything, that does not make sense.");
			}

			throw new InvalidOperationException($"Cannot discard given components {components}, discard is invalid and probably does not make sense.");
		}

		protected abstract TNetwork CreateNetwork(string name);

		protected abstract INetworkSelector<TNetwork> CreateSelector(TNetwork network);
	}
}
