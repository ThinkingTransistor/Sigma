using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;

namespace Sigma.Core.Training.Mergers
{
	/// <summary>
	/// This network mergers allows the use of multiple different network mergers.
	/// </summary>
	public class MultiNetworkMerger : INetworkMerger
	{
		/// <summary>
		/// The network mergers that will be applied.
		/// </summary>
		public INetworkMerger[] Mergers { get; }

		/// <summary>
		/// Create a multi network merger with a set of other mergers.
		/// </summary>
		/// <param name="mergers">All mergers that will be applied. May not be <c>null</c> nor empty.</param>
		public MultiNetworkMerger(params INetworkMerger[] mergers)
		{
			if (mergers == null) throw new ArgumentNullException(nameof(mergers));
			if (mergers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(mergers));

			Mergers = mergers;
		}

		/// <summary>
		///     Specify how multiple networks are merged into a single one. <see ref="root" /> is <em>not</em>
		///     considered for the calculation. It is merely the storage container. (Although root can also be in
		///     <see ref="networks" />).
		/// </summary>
		/// <param name="root">
		///     The root network that will be modified. Since the <see cref="INetworkMerger" /> does not know how
		///     to create a <see ref="INetwork" />, it will be passed not returned.
		/// </param>
		/// <param name="networks">
		///     The networks that will be merged into the <see ref="root" />. Can contain <see ref="root" />
		///     itself.
		/// </param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		public void Merge(INetwork root, IEnumerable<INetwork> networks, IComputationHandler handler = null)
		{
			IEnumerable<INetwork> networksEnumerable = networks as INetwork[] ?? networks.ToArray();

			foreach (INetworkMerger networkMerger in Mergers)
			{
				networkMerger.Merge(root, networksEnumerable, handler);
			}
		}

		/// <summary>
		///     Specify how two networks are merged into a single one (root = root + other). This method can be achieved with
		///     <see cref="INetworkMerger.Merge(INetwork,IEnumerable{INetwork},IComputationHandler)"></see>
		/// ,
		///     but it may be easier to call this function.
		/// </summary>
		/// <param name="root">The root network that will be modified and contain the result.</param>
		/// <param name="other">The unchanged other network.</param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		public void Merge(INetwork root, INetwork other, IComputationHandler handler = null)
		{
			foreach (INetworkMerger networkMerger in Mergers) { networkMerger.Merge(root, other, handler); }
		}

		/// <summary>
		///     Specify the registry keys (match identifiers) that will be merged.
		///     This supports the full
		///     <see cref="Utils.IRegistryResolver" /> syntax.
		/// </summary>
		/// <param name="matchIdentifier">The key of the registry.</param>
		public void AddMergeEntry(string matchIdentifier)
		{
			throw new InvalidOperationException("Merge entries can not be added to a multi merger. Add them to the individual networks.");
		}

		/// <summary>
		///     Remove a previously specified match identifier from all networks.
		/// </summary>
		/// <param name="matchIdentifier">The key to remove from the targeted registries.</param>
		public void RemoveMergeEntry(string matchIdentifier)
		{
			foreach (INetworkMerger networkMerger in Mergers)
			{
				networkMerger.RemoveMergeEntry(matchIdentifier);
			}
		}
	}
}