/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;

namespace Sigma.Core.Persistence.Selectors
{
	/// <summary>
	/// A network selector to selectively keep and discard <see cref="INetwork"/> data.
	/// </summary>
	/// <typeparam name="T">The network type.</typeparam>
	public interface INetworkSelector<out T> : ISelector<T> where T : INetwork
	{
		/// <summary>
		/// Keep a set of network components, discard all other components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to keep.</param>
		/// <returns>A selector for a new network with the given component(s) retained.</returns>
		ISelector<T> Keep(params TrainerComponent[] components);

		/// <summary>
		/// Discard specified network components in a new network.
		/// </summary>
		/// <param name="components">The component(s) to discard.</param>
		/// <returns>A selector for a new network with the given component(s) discarded.</returns>
		ISelector<T> Discard(params TrainerComponent[] components);
	}

	/// <summary>
	/// An individual network component. 
	/// </summary>
	public class NetworkComponent : SelectorComponent
	{
		/// <summary>
		/// Create a selector component with a certain id. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		public NetworkComponent(int id) : base(id)
		{
		}

		/// <summary>
		/// Create a selector component with a certain id and certain sub-components. 
		/// The id must be the same for semantically equivalent components at the current level
		///  (i.e. a trainer component must have the same id as another trainer component with different sub-flags).
		/// </summary>
		/// <param name="id">The id.</param>
		/// <param name="subComponents">The selected sub components.</param>
		public NetworkComponent(int id, params SelectorComponent[] subComponents) : base(id, subComponents)
		{
		}

		/// <summary>
		/// Nothing (except the environment name, which is the minimum state and included by default).
		/// </summary>
		public static readonly NetworkComponent None = new NetworkComponent(0);

		/// <summary>
		/// The <see cref="INetworkArchitecture"/> of this model. 
		/// </summary>
		public static readonly NetworkComponent Architecture = new NetworkComponent(0);

		/// <summary>
		/// Everything.
		/// </summary>
		public static readonly NetworkComponent Everything = new NetworkComponent(int.MaxValue);
	}
}
