/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Workers
{
	/// <summary>
	///     A single worker which directly executes the training process defined in a trainer and is spawned by an operator to
	///     complete some part of a training task.
	/// 
	///     Typically multiple workers are used simultaneously and then their individual copies of the trained models are
	///     merged at certain intervals for optimal device usage.
	/// </summary>
	public interface IWorker
	{
		/// <summary>
		///     The operator which controls this worker.
		/// </summary>
		IOperator Operator { get; }

		/// <summary>
		///     The current state of the worker. <see cref="ExecutionState.None" />
		///     if the worker has not been started yet;
		/// </summary>
		ExecutionState State { get; }

		/// <summary>
		///     The computation handler to use for computation and ndarray management.
		/// </summary>
		IComputationHandler Handler { get; }

		/// <summary>
		///		A local copy of the global training data iterator. Used to enable parallel network training.
		/// </summary>
		IDataIterator LocalTrainingDataIterator { get; set; }

		/// <summary>
		///		A local copy of the global optimiser. Used to enable parallel network training.
		/// </summary>
		IOptimiser LocalOptimiser { get; set; }

		/// <summary>
		///     A local copy of the network (model) to train. Used to enable parallel network training.
		/// </summary>
		INetwork LocalNetwork { get; set; }

		/// <summary>
		///     The current epoch number (i.e. how many epochs have been executed on this worker).
		/// </summary>
		int LocalEpochNumber { get; }

		/// <summary>
		///		The iteration number within the current epoch (i.e. how many training iterations have been executed on this worker in the current epoch).
		/// </summary>
		int LocalIterationNumber { get; }

		/// <summary>
		///     Start this worker and start training the network as defined in the trainer and ordered by the operator.
		/// </summary>
		void Start();

		/// <summary>
		///		Start this worker for one iteration with the parameters defined in the trainer.
		///		All the initialisation happens here. 
		/// </summary>
		void RunOnce();

		/// <summary>
		///     Signal this worker to pause at the next opportunity (after an iteration).
		/// </summary>
		void SignalPause();

		/// <summary>
		///     Signal this worker to resume the work.
		/// </summary>
		void SignalResume();

		/// <summary>
		///     Signal this worker to stop the execution as soon as possible.
		/// </summary>
		void SignalStop();

		/// <summary>
		/// Eject a certain time scale event and invoke the active hooks that correspond to the local time scale change.
		/// </summary>
		/// <param name="timeScale">The time scale.</param>
		void EjectAndInvokeTimeScaleEvent(TimeScale timeScale);
	}
}