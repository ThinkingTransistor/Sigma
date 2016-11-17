/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Training.Operators.Workers
{
	/// <summary>
	/// A single worker which directly executes the training process defined in a trainer and is spawned by an operator to complete some part of a training task.
	/// Typically multiple workers are used simultaneously and then their individual copies of the trained models are merged at certain intervals for optimal device usage.
	/// </summary>
	public interface IWorker
	{
		/// <summary>
		/// The operator which controls this worker. 
		/// </summary>
		IOperator Operator { get; }

		/// <summary>
		/// The computation handler to use for computation and ndarray management. 
		/// </summary>
		IComputationHandler Handler { get; }

		/// <summary>
		/// A local copy of the network (model) to train. Used to enable parallel network training. 
		/// </summary>
		INetwork LocalNetwork { get; }

		/// <summary>
		/// The current iteration number since last synchronisation (i.e. how many iterations have been executed on this worker).
		/// </summary>
		int LocalIterationNumber { get; }

		/// <summary>
		/// Start this worker and start training the network as defined in the trainer and ordered by the operator. 
		/// </summary>
		void Start();

		/// <summary>
		/// Signal this worker to pause at the next opportunity (after an iteration). 
		/// </summary>
		void SignalPause();
	}
}
