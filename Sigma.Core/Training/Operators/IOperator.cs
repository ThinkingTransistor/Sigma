/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators
{
	/// <summary>
	///     An operator that operates (executes) the training process defined in a trainer.
	///     Operators typically split the workload into multiple workers and backends for CPU, GPU and inter-device cooperation
	///     are provided.
	/// </summary>
	public interface IOperator
	{
		/// <summary>
		///     The <see cref="SigmaEnvironment" /> this operator runs in and communicates with.
		///     It will be automatically set by the <see cref="ITrainer" />.
		/// </summary>
		SigmaEnvironment Sigma { get; set; }

		/// <summary>
		///		A registry containing relevant parameters of this operator.
		/// </summary>
		IRegistry Registry { get; }

		/// <summary>
		///     The current <see cref="ExecutionState" /> of the <see cref="IOperator" />. <see cref="ExecutionState.None" />
		///     if the operator has not been started yet.
		/// </summary>
		ExecutionState State { get; }

		/// <summary>
		///     The <see cref="IComputationHandler" /> used to compute everything in
		///     this <see cref="IOperator" />. It will be automatically set by the
		///     <see cref="ITrainer" /> if not specified.
		/// </summary>
		IComputationHandler Handler { get; set; }

		/// <summary>
		///     The <see cref="ITrainer" /> that is being trained in this operators training process.
		///     This is automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		ITrainer Trainer { get; set; }

		/// <summary>
		///     The <see cref="INetwork" /> the training process is operated on.
		///     This is automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		INetwork Network { get; set; }

		/// <summary>
		///		This merger is used to merge multiple networks after they are
		///		submitted to the <see cref="IOperator"/>.
		/// </summary>
		INetworkMerger Merger { get; set; }

		/// <summary>
		///     The number of <see cref="Workers.IWorker" />s (threads) used in this
		///     <see cref="IOperator" /> in parallel.
		/// </summary>
		int WorkerCount { get; }

		/// <summary>
		///		The number of the current global epoch in this operator.
		/// </summary>
		int EpochNumber { get; }

		/// <summary>
		///     Attach an active hook to this operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		void AttachHook(IActiveHook hook);

		/// <summary>
		///     Attach a passive hook to this operator.
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		void AttachHook(IPassiveHook hook);

		/// <summary>
		///     Detach a hook from this operator.
		/// </summary>
		/// <param name="hook">The hook to detach.</param>
		void DetachHook(IActiveHook hook);

		/// <summary>
		///     Detach a passive from this operator.
		/// </summary>
		/// <param name="hook">The hook to detach.</param>
		void DetachHook(IPassiveHook hook);

		/// <summary>
		/// Mark a hook as dead in a certain worker.
		/// </summary>
		/// <param name="hook">The hook to mark.</param>
		/// <param name="worker">The worker in which this hook was deemed dead.</param>
		void MarkHookDead(IActiveHook hook, IWorker worker);

		/// <summary>
		/// Invoke hooks for a certain time scale with a certain worker.
		/// </summary>
		/// <param name="timeScale">The time scale.</param>
		/// <param name="worker">The worker to invoke the hook with.</param>
		/// <param name="hooks">The hooks to check and invoke.</param>
		/// <param name="localHookTimeSteps">The local hook time steps to use (and populate if missing).</param>
		/// <param name="resultHooksToInvoke"></param>
		void EjectTimeScaleEvent(TimeScale timeScale, IWorker worker, IEnumerable<IHook> hooks, IDictionary<IHook, ITimeStep> localHookTimeSteps, ISet<IHook> resultHooksToInvoke);

		/// <summary>
		///     Push the workers current progress (e.g. local network) to the <see cref="IOperator"/>. 
		/// </summary>
		/// <param name="worker">The worker.</param>
		void PushProgress(IWorker worker);

		/// <summary>
		///     Pull the progress of the <see cref="IOperator"/> to the worker (e.g. copy of global network).
		/// </summary>
		/// <param name="worker">The worker.</param>
		void PullProgress(IWorker worker);

		/// <summary>
		///     Start this operator in a separate thread (return immediately).
		/// </summary>
		void Start();

		/// <summary>
		///		Start this operator for a single time only (return immediately).
		/// </summary>
		void StartOnce();

		/// <summary>
		///     Signal this operator to pause as soon as possible.
		/// </summary>
		void SignalPause();

		/// <summary>
		///     Signal this operator to resume as soon as possible.
		/// </summary>
		void SignalResume();

		/// <summary>
		///     Signal this operator to stop as soon as possible.
		/// </summary>
		void SignalStop();

		/// <summary>
		///     This method blocks until the last state change has been fully performed.
		///     Returns immediately if not implemented.
		/// </summary>
		void WaitForStateChanged();

		/// <summary>
		///		Populate a registry using a certain worker's local values.
		/// </summary>
		/// <param name="registry">The registry to populate.</param>
		/// <param name="worker">The worker to fetch local values from.</param>
		void PopulateWorkerRegistry(IRegistry registry, IWorker worker);
	}
}