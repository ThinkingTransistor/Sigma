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
	public interface IOptimiser : IDeepCopyable // TODO the deep copy methods are currently implemented as shallow copies and kind of abused because workers get local shallow optimiser copies
	{
		/// <summary>
		/// The registry containing data about this optimiser and its last run.
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		/// Prepare for a single iteration of the network (model) optimisation process (<see cref="Run"/>).
		/// Typically used to trace trainable parameters to retrieve the derivatives in <see cref="Run"/>.
		/// </summary>
		/// <param name="network">The network to prepare for optimisation.</param>
		/// <param name="handler">THe handler to use.</param>
		void PrepareRun(INetwork network, IComputationHandler handler);

		/// <summary>
		/// Run a single iteration of the network (model) optimisation process (e.g. backward pass only). 
		/// Note: The gradients are typically used to update the parameters in a certain way to optimise the network.
		/// </summary>
		/// <param name="network">The network to optimise.</param>
		/// <param name="handler">The computation handler to use.</param>
		void Run(INetwork network, IComputationHandler handler);

		/// <summary>
		/// Add a specific filter mask ("freeze" a specific part of the model).
		/// Filter masks are registry resolve strings for the model, e.g. layer1.*, *.weights.
		/// </summary>
		/// <param name="filterMask">The filter mask to add ("freeze").</param>
		void AddFilter(string filterMask);

		/// <summary>
		/// Remove a specific filter mask ("unfreeze" a specific part of the model).
		/// </summary>
		/// <param name="filterMask">The filter mask to remove ("unfreeze").</param>
		void RemoveFilter(string filterMask);

		/// <summary>
		/// Clear all existing filter masks ("unfreeze" the entire model).
		/// </summary>
		void ClearFilters();
	}
}
