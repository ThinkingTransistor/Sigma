/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Utils;

namespace Sigma.Core.Architecture
{
	/// <summary>
	/// A network architecture from which an actual network can be constructed.
	/// </summary>
	public interface INetworkArchitecture : IDeepCopyable
	{
		/// <summary>
		/// The total number of layers in this architecture.
		/// </summary>
		int LayerCount { get; }

		/// <summary>
		/// A registry containing all relevant parameters and sub-registries (e.g. layer constructs).
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Validate this network architecture. 
		/// </summary>
		void Validate();

		/// <summary>
		/// Resolve all layer names to be fully qualified.
		/// Note: Unresolved names are stored to enable consistency when changing the architecture and re-resolving the layer names.
		/// </summary>
		void ResolveAllNames();

		/// <summary>
		/// Yield all layers in the order they should be processed. 
		/// </summary>
		/// <returns></returns>
		IEnumerable<LayerConstruct> YieldLayerConstructsOrdered();
	}
}
