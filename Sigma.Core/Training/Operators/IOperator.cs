/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Persistence.Selectors;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;
using System.Collections.Generic;

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
		/// The total running time of this operator since start in milliseconds (running only when the <see cref="ExecutionState"/> is <see cref="ExecutionState.Running"/>).
		/// </summary>
		long RunningTimeMilliseconds { get; }

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
		INetworkMerger NetworkMerger { get; set; }

		/// <summary>
		///		The local hooks that are attached to this operator.
		/// </summary>
		IReadOnlyCollection<IHook> AttachedLocalHooks { get; }

		/// <summary>
		///		The global hooks that are attached to this operator.
		/// </summary>
		IReadOnlyCollection<IHook> AttachedGlobalHooks { get; }

		/// <summary>
		///		All local hooks sorted by time scale.
		/// </summary>
		IReadOnlyDictionary<TimeScale, ISet<IHook>> AttachedLocalHooksByTimeScale { get; }

		/// <summary>
		///		All global hooks sorted by time scale.
		/// </summary>
		IReadOnlyDictionary<TimeScale, ISet<IHook>> AttachedGlobalHooksByTimescale { get; }

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
		///		Enable the use of sessions in this operator. 
		///		Computation handlers will attempt to speed up computation sessions at the possible expense of higher memory requirements.
		/// </summary>
		bool UseSessions { get; set; }

		/// <summary>
		///     Attach a local hook to this operator (if no functionally equivalent hook is already attached).
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		/// <returns>A boolean indicating if the hook was successfully attached.</returns>
		bool AttachLocalHook(IHook hook);

		/// <summary>
		///     Attach a global hook to this operator (if no functionally equivalent hook is already attached).
		/// </summary>
		/// <param name="hook">The hook to attach.</param>
		/// <returns>A boolean indicating if the hook was successfully attached.</returns>
		bool AttachGlobalHook(IHook hook);

		/// <summary>
		///     Detach a local hook from this operator.
		/// </summary>
		/// <param name="hook">The hook to detach.</param>
		/// <returns>A boolean indicating if the hook was successfully detached.</returns>
		bool DetachLocalHook(IHook hook);

		/// <summary>
		///     Detach a global from this operator.
		/// </summary>
		/// <param name="hook">The hook to detach.</param>
		/// <returns>A boolean indicating if the hook was successfully detached.</returns>
		bool DetachGlobalHook(IHook hook);

		/// <summary>
		/// Mark a (local) hook as dead in a certain worker.
		/// </summary>
		/// <param name="hook">The hook to mark.</param>
		/// <param name="worker">The worker in which this hook was deemed dead.</param>
		void MarkHookDead(IHook hook, IWorker worker);

		/// <summary>
		/// Get the invocation index for a certain local hook. 
		/// This invocation index represents the index at which this operator should be invoked.
		/// Used for ordering hooks to satisfy all dependencies upon invocation.
		/// Note: All hooks with a smaller invocation index and the same invocation target should be invoked before this hook.
		/// </summary>
		/// <param name="hook">The hook.</param>
		/// <returns>The invocation index of the given local hook.</returns>
		uint GetLocalHookInvocationIndex(IHook hook);

		/// <summary>
		/// Get the invocation target for a certain local hook.
		/// The invocation target represents the thread in which the hook should be invoked.
		/// Used for putting background hooks with dependencies in the right "invocation bucket" for dependency satisfaction.
		/// Note:   Only background hooks require invocation targets.
		///			The invocation target of a foreground hook is implicitly the owning thread. 
		/// </summary>
		/// <param name="hook">The hook.</param>
		/// <returns>The invocation target for the given local hook.</returns>
		uint GetLocalHookInvocationTarget(IHook hook);

		/// <summary>
		/// Get the invocation index for a certain global hook. 
		/// This invocation index represents the index at which this operator should be invoked.
		/// Used for ordering hooks to satisfy all dependencies upon invocation.
		/// Note: All hooks with a smaller invocation index and the same invocation target should be invoked before this hook.
		/// </summary>
		/// <param name="hook">The hook.</param>
		/// <returns>The invocation index of the given global hook.</returns>
		uint GetGlobalHookInvocationIndex(IHook hook);

		/// <summary>
		/// Get the invocation target for a certain global hook.
		/// The invocation target represents the thread in which the hook should be invoked.
		/// Used for putting background hooks with dependencies in the right "invocation bucket" for dependency satisfaction.
		/// Note:   Only background hooks require invocation targets.
		///			The invocation target of a foreground hook is implicitly the owning thread. 
		/// </summary>
		/// <param name="hook">The hook.</param>
		/// <returns>The invocation target for the given global hook.</returns>
		uint GetGlobalHookInvocationTarget(IHook hook);

		/// <summary>
		/// Execute a given command. It is uncertain when the command is executed.
		/// </summary>
		/// <param name="command">The <see cref="ICommand"/> that will be executed.</param>
		void InvokeCommand(ICommand command);

		/// <summary>
		/// Dispatch a set of hooks for background invocation. The required registry entries are automatically copied from the given local registry. 
		/// </summary>
		/// <param name="hooksToInvokeInBackground">The hooks to invoke in the background.</param>
		/// <param name="localRegistry">The local registry to copy required registry entries from.</param>
		/// <param name="bufferRegistryEntries">The buffer for fetching required registry entries.</param>
		/// <param name="bufferResolvedRegistryEntries">The buffer for resolved registry entries.</param>
		void DispatchBackgroundHookInvocation(IList<IHook> hooksToInvokeInBackground, IRegistry localRegistry, ISet<string> bufferRegistryEntries, ISet<string> bufferResolvedRegistryEntries);

		/// <summary>
		/// Invoke hooks for a certain time scale with a certain worker.
		/// </summary>
		/// <param name="timeScale">The time scale.</param>
		/// <param name="hooksByTimescale">The hooks to check and invoke (by timescale).</param>
		/// <param name="localHookTimeSteps">The local hook time steps to use (and populate if missing).</param>
		/// <param name="resultHooksToInvoke">The resulting (unordered) hooks to invoke.</param>
		void EjectTimeScaleEvent(TimeScale timeScale, IReadOnlyDictionary<TimeScale, ISet<IHook>> hooksByTimescale, IDictionary<IHook, ITimeStep> localHookTimeSteps, List<IHook> resultHooksToInvoke);

		/// <summary>
		///     Push the workers current progress (e.g. local network) to the <see cref="IOperator"/>. 
		///		Note: The operator determines what parts of the progress to push and use (e.g. depending on local / global iteration / epoch).
		/// </summary>
		/// <param name="worker">The worker.</param>
		void PushProgress(IWorker worker);

		/// <summary>
		///     Pull the progress of the <see cref="IOperator"/> to the worker (e.g. copy of global network) if a newer version is available.
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
		///		Signal this operator to reset as soon as possible.
		///     This operator will be reset to the initial state (runtime data is discarded, network remains untouched, workers are kept).
		/// </summary>
		void SignalReset();

		/// <summary>
		///     This method blocks until the last state change has been fully performed.
		///     Returns immediately if not implemented.
		/// </summary>
		void WaitForStateChanged();

		/// <summary>
		///     This method blocks until the last worker state change has been fully performed.
		///     Returns immediately if not implemented.
		/// </summary>
		void WaitForWorkersStateChanged();

		/// <summary>
		///		Populate a registry using a certain worker's local values.
		/// </summary>
		/// <param name="registry">The registry to populate.</param>
		/// <param name="worker">The worker to fetch local values from.</param>
		void PopulateWorkerRegistry(IRegistry registry, IWorker worker);

		/// <summary>
		/// Get a shallow copy of this operator, including all available runtime state.
		/// </summary>
		/// <returns></returns>
		IOperator ShallowCopy();

		/// <summary>
		/// Get an operator selector for this operator.
		/// </summary>
		/// <returns>The selector for this operator.</returns>
		IOperatorSelector<IOperator> Select();
	}
}