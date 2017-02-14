/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;

namespace Sigma.Core.Training.Mergers
{
	/// <summary>
	///     This interface specifies a way to merge multiple <see cref="INetwork" />s.
	///     When multiple <see cref="Operators.Workers.IWorker" /> execute their tasks,
	///     they push their current network to an <see cref="Operators.IOperator" />. These
	///     pushed networks have to be merged by an <see cref="INetworkMerger" />.
	/// </summary>
	public interface INetworkMerger
	{
		/// <summary>
		///     Specify how multiple networks are merged into a single one. <see cref="root" /> is <em>not</em>
		///     considered for the calculation. It is merely the storage container. (Although root can also be in
		///     <see cref="networks" />).
		/// </summary>
		/// <param name="root">
		///     The root network that will be modified. Since the <see cref="INetworkMerger" /> does not know how
		///     to create a <see cref="INetwork" />, it will be passed not returned.
		/// </param>
		/// <param name="networks">
		///     The networks that will be merged into the <see cref="root" />. Can contain <see cref="root" />
		///     itself.
		/// </param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		void Merge(INetwork root, IEnumerable<INetwork> networks, IComputationHandler handler = null);

		/// <summary>
		///     Specify how two networks are merged into a single one (root = root + other). This method can be achieved with
		///     <see cref="Merge(INetwork,IEnumerable{INetwork},IComputationHandler)" />,
		///     but it may be easier to call this function.
		/// </summary>
		/// <param name="root">The root network that will be modified and contain the result.</param>
		/// <param name="other">The unchanged other network.</param>
		/// <param name="handler">
		///     A handler can be specified optionally. If not passed (but required),
		///     <see cref="MathAbstract.ITraceable.AssociatedHandler" /> will be used.
		/// </param>
		void Merge(INetwork root, INetwork other, IComputationHandler handler = null);

		/// <summary>
		///     Specify the registry keys (match identifiers) that will be merged.
		///     This supports the full
		///     <see cref="Utils.IRegistryResolver" /> syntax.
		/// </summary>
		/// <param name="matchIdentifier">The key of the registry.</param>
		void AddMergeEntry(string matchIdentifier);

		/// <summary>
		///     Remove a previously specified match identifier.
		/// </summary>
		/// <param name="matchIdentifier">The key to remove from the targeted registries.</param>
		void RemoveMergeEntry(string matchIdentifier);
	}
}