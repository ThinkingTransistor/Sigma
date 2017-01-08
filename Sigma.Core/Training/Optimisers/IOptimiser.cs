/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Optimisers
{
	/// <summary>
	/// An optimiser that defines how a network (model) should be optimised by implementing a single iteration (forward and backward pass).
	/// </summary>
	public interface IOptimiser : IDeepCopyable
	{
		/// <summary>
		/// The registry containing data about this optimiser and its last run.
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Run a single iteration of the network (model) optimisation process (forward and backward pass). 
		/// Note: The gradients are typically used to update the parameters in a certain way to optimise the network.
		/// </summary>
		/// <param name="network">The network to optimise.</param>
		/// <param name="handler">The computation handler to use.</param>
		void Run(INetwork network, IComputationHandler handler);
	}
}
