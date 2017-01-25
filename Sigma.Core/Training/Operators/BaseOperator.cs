/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using log4net;
using static Sigma.Core.Utils.ThreadUtils;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators
{
	[Serializable]
	public abstract class BaseOperator : IOperator
	{
		/// <summary>
		///		A registry containing relevant parameters of this operator.
		/// </summary>
		public IRegistry Registry { get; }

		/// <summary>
		///     All <see cref="IActiveHook" />s that are attached to this <see cref="IOperator" />.
		/// </summary>
		protected readonly ICollection<IActiveHook> ActiveHooks;

		/// <summary>
		///     All <see cref="IPassiveHook" />s that are attached to this <see cref="IOperator" />.
		/// </summary>
		protected readonly ICollection<IPassiveHook> PassiveHooks;

		/// <summary>
		///		All active hooks sorted by time scale.
		/// </summary>
		protected readonly IDictionary<TimeScale, ISet<IActiveHook>> ActiveHooksByTimeScale;

		/// <summary>
		///		All passive hooks sorted by time scale.
		/// </summary>
		protected readonly IDictionary<TimeScale, ISet<IPassiveHook>> PassiveHooksByTimescale;

		/// <summary>
		///		The alive hooks by an array of flags of workers keeping it alive.
		/// </summary>
		protected readonly IDictionary<IActiveHook, bool[]> AliveHooksByInWorkerStates;

		/// <summary>
		///     All the <see cref="IWorker" />s managed by this operator.
		/// </summary>
		protected IEnumerable<IWorker> Workers;

		/// <summary>
		///		The worker indices by workers for quick access.
		/// </summary>
		protected IReadOnlyDictionary<IWorker, int> WorkerIndicesByWorkers;

		/// <summary>
		///     The <see cref="SigmaEnvironment" /> this operator runs in and communicates with.
		///     It will be automatically set by the <see cref="ITrainer" />.
		/// </summary>
		public SigmaEnvironment Sigma { get; set; }

		/// <summary>
		///     The current <see cref="ExecutionState" /> of the <see cref="IOperator" />. <see cref="ExecutionState.None" />
		///     if the operator has not been started yet.
		/// </summary>
		public ExecutionState State { get; protected set; } = ExecutionState.None;

		/// <summary>
		///     The <see cref="IComputationHandler" /> used to compute everything in
		///     this <see cref="IOperator" />. It will be automatically set by the
		///     <see cref="ITrainer" /> if not specified.
		/// </summary>
		public IComputationHandler Handler { get; set; }

		/// <summary>
		///     The <see cref="ITrainer" /> that is being trained in this operators training process.
		///     This will be automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		public ITrainer Trainer { get; set; }

		/// <summary>
		///     The <see cref="INetwork" /> the training process is operated on.
		///     This will be automatically set by the corresponding <see cref="ITrainer" />.
		/// </summary>
		public INetwork Network { get; set; }

		/// <summary>
		///		This merger is used to merge multiple networks after they get
		///		reported to the <see cref="IOperator"/>. Defaults to <see cref="AverageNetworkMerger"/>.
		/// </summary>
		public INetworkMerger NetworkMerger { get; set; } = new AverageNetworkMerger();

		/// <summary>
		///     The number of <see cref="IWorker" />s (threads) used in this
		///     <see cref="IOperator" /> in parallel.
		/// </summary>
		public int WorkerCount { get; }

		/// <summary>
		///		The number of the current global epoch in this operator.
		/// </summary>
		public int EpochNumber { get; protected set; }

		/// <summary>
		/// The logger, it will be initialised in the property so that the class matches.
		/// </summary>
		private ILog _logger;

		/// <summary>
		/// The logger for the inherited class. 
		/// </summary>
		protected ILog Logger => _logger ?? (_logger = LogManager.GetLogger(GetType()));

		/// <summary>
		/// The lock that will be used to perform asynchronous management of the <see cref="IWorker"/>.
		/// </summary>
		private readonly object _stateChangeLock;

		private readonly IRegistryResolver _bufferRegistryResolver;
		private readonly ISet<string> _bufferCurrentRequiredHookParameters;
		private readonly ISet<string> _bufferPreviousRequiredHookParameters;
		private readonly ISet<string> _bufferResolvedRequiredHookParameters;

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> using the default <see cref="IComputationHandler" /> (currently <see cref="CpuFloat32Handler"/>.
		///     The <see cref="IComputationHandler" /> will be automatically set by the <see cref="ITrainer" />.
		///		TODO update documentation 
		/// </summary>
		/// <param name="workerCount">
		///     The number of <see cref="IWorker" />s (threads) used in this <see cref="IOperator" /> in
		///     parallel.
		/// </param>
		protected BaseOperator(int workerCount) : this(new CpuFloat32Handler(), workerCount)
		{
		}

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> with a specified <see cref="IComputationHandler" />.
		///     The <see cref="IComputationHandler" /> will <c>not</c> be modified by the <see cref="ITrainer" />.
		/// </summary>
		/// <param name="handler">
		///     The <see cref="IComputationHandler" /> that will be assigned to the
		///     <see cref="IComputationHandler" />
		/// </param>
		/// <param name="workerCount">
		///     The number of <see cref="IWorker" />s (threads) used in this <see cref="IOperator" /> in
		///     parallel.
		/// </param>
		protected BaseOperator(IComputationHandler handler, int workerCount)
		{
			if (handler == null) throw new ArgumentNullException(nameof(handler));

			_stateChangeLock = new object();

			Handler = handler;
			WorkerCount = workerCount;
			EpochNumber = -1;

			ActiveHooks = new List<IActiveHook>();
			PassiveHooks = new List<IPassiveHook>();
			ActiveHooksByTimeScale = new Dictionary<TimeScale, ISet<IActiveHook>>();
			PassiveHooksByTimescale = new Dictionary<TimeScale, ISet<IPassiveHook>>();
			AliveHooksByInWorkerStates = new Dictionary<IActiveHook, bool[]>();

			Registry = new Registry(tags: "operator");
			_bufferRegistryResolver = new RegistryResolver(Registry);

			_bufferCurrentRequiredHookParameters = new HashSet<string>();
			_bufferPreviousRequiredHookParameters = new HashSet<string>();
			_bufferResolvedRequiredHookParameters = new HashSet<string>();
		}

		public void AttachHook(IActiveHook hook)
		{
			if (ActiveHooks.Contains(hook))
			{
				Logger.Debug($"Unable to attach active hook {hook} to operator {this}, hook is already attached.");

				return;
			}

			ActiveHooks.Add(hook);

			if (!ActiveHooksByTimeScale.ContainsKey(hook.TimeStep.TimeScale))
			{
				ActiveHooksByTimeScale.Add(hook.TimeStep.TimeScale, new HashSet<IActiveHook>());
			}

			ActiveHooksByTimeScale[hook.TimeStep.TimeScale].Add(hook);

			AliveHooksByInWorkerStates.Add(hook, new bool[WorkerCount].Populate(true));

			Logger.Debug($"Attached active hook {hook} to operator {this}.");
		}

		public void DetachHook(IActiveHook hook)
		{
			if (ActiveHooks.Remove(hook))
			{
				ActiveHooksByTimeScale[hook.TimeStep.TimeScale].Remove(hook);
				AliveHooksByInWorkerStates.Remove(hook);

				Logger.Debug($"Detached active hook {hook} from operator {this}.");
			}
		}

		public void AttachHook(IPassiveHook hook)
		{
			if (PassiveHooks.Contains(hook))
			{
				Logger.Debug($"Unable to attach passive hook {hook} to operator {this}, hook is already attached.");

				return;
			}

			PassiveHooks.Add(hook);

			if (!PassiveHooksByTimescale.ContainsKey(hook.TimeStep.TimeScale))
			{
				PassiveHooksByTimescale.Add(hook.TimeStep.TimeScale, new HashSet<IPassiveHook>());
			}

			PassiveHooksByTimescale[hook.TimeStep.TimeScale].Add(hook);

			Logger.Debug($"Attached passive hook {hook} to operator {this}.");
		}

		public void DetachHook(IPassiveHook hook)
		{
			if (PassiveHooks.Remove(hook))
			{
				PassiveHooksByTimescale[hook.TimeStep.TimeScale].Remove(hook);

				Logger.Debug($"Detached passive hook {hook} from operator {this}");
			}
		}

		/// <summary>
		/// Mark an active hook as dead in a certain worker.
		/// </summary>
		/// <param name="hook">The hook to mark.</param>
		/// <param name="worker">The worker in which this hook was deemed dead.</param>
		public void MarkHookDead(IActiveHook hook, IWorker worker)
		{
			if (!AliveHooksByInWorkerStates.ContainsKey(hook))
			{
				throw new InvalidOperationException($"Unable to mark hook {hook} as dead in operator {this} for worker {worker}, hook is not registered as alive.");
			}

			if (!WorkerIndicesByWorkers.ContainsKey(worker))
			{
				throw new InvalidOperationException($"Unable to mark hook {hook} as dead in operator {this} for worker {worker}, worker does not belong to this operator.");
			}

			bool[] aliveFlags = AliveHooksByInWorkerStates[hook];

			aliveFlags[WorkerIndicesByWorkers[worker]] = false;

			if (aliveFlags.All(flag => !flag))
			{
				Logger.Debug($"Detaching hook {hook} in operator {this}, hook is deemed completely dead and can be safely detached.");

				DetachHook(hook);
			}
		}

		protected void ResolveAllRequiredRegistryEntries(IRegistryResolver registryResolver, ISet<string> allRequiredRegistryEntries, ISet<string> resultAllResolvedRequiredRegistryEntries)
		{
			resultAllResolvedRequiredRegistryEntries.Clear();

			foreach (string registryEntry in allRequiredRegistryEntries)
			{
				string[] resolvedEntries;

				registryResolver.ResolveGet<object>(registryEntry, out resolvedEntries, null);
			}
		}

		protected void FetchAllRequiredRegistryEntries(IEnumerable<IHook> hooks, ISet<string> bufferAllRequiredRegistryEntries = null)
		{
			if (bufferAllRequiredRegistryEntries == null)
			{
				bufferAllRequiredRegistryEntries = new HashSet<string>();
			}
			else
			{
				bufferAllRequiredRegistryEntries.Clear();
			}

			foreach (IHook hook in hooks)
			{
				foreach (string registryEntry in hook.RequiredRegistryEntries)
				{
					if (!bufferAllRequiredRegistryEntries.Contains(registryEntry))
					{
						bufferAllRequiredRegistryEntries.Add(registryEntry);
					}
				}
			}
		}

		/// <summary>
		/// Eject a certain time scale event within a certain worker and update the local time steps.
		/// </summary>
		/// <param name="timeScale">The time scale.</param>
		/// <param name="worker">The worker to invoke the hook with.</param>
		/// <param name="hooks">The hooks to check and invoke.</param>
		/// <param name="localHookTimeSteps">The local hook time steps to use (and populate if missing).</param>
		/// <param name="resultHooksToInvoke">The resulting hooks to invoke.</param>
		public void EjectTimeScaleEvent(TimeScale timeScale, IWorker worker, IEnumerable<IHook> hooks, IDictionary<IHook, ITimeStep> localHookTimeSteps, ISet<IHook> resultHooksToInvoke)
		{
			if (timeScale == null) throw new ArgumentNullException(nameof(timeScale));
			if (worker == null) throw new ArgumentNullException(nameof(worker));
			if (hooks == null) throw new ArgumentNullException(nameof(hooks));
			if (localHookTimeSteps == null) throw new ArgumentNullException(nameof(localHookTimeSteps));
			if (resultHooksToInvoke == null) throw new ArgumentNullException(nameof(resultHooksToInvoke));

			Logger.Debug($"Ejecting time scale event {timeScale} for worker {worker} in operator {this}...");

			resultHooksToInvoke.Clear();

			foreach (IHook hook in hooks)
			{
				if (hook.TimeStep.LocalLiveTime == 0)
				{
					continue;
				}

				if (!localHookTimeSteps.ContainsKey(hook))
				{
					TimeStep timeStep = (TimeStep) hook.TimeStep.DeepCopy();

					timeStep.LocalLiveTime = timeStep.LiveTime;
					timeStep.LocalInterval = timeStep.Interval;
				}

				ITimeStep localTimeStep = localHookTimeSteps[hook];

				localTimeStep.LocalInterval--;

				if (localTimeStep.LocalInterval == 0)
				{
					// TODO invoke hook / return hooks to invoke somehow? -> add to worker
					resultHooksToInvoke.Add(hook);

					if (localTimeStep.LocalLiveTime > 0)
					{
						localTimeStep.LocalLiveTime--;

						//TODO check if LocalLiveTime == 0 in callee and mark as dead accordingly
					}

					localTimeStep.LocalInterval = localTimeStep.Interval;
				}
			}

			Logger.Debug($"Done ejecting time scale event {timeScale} for worker {worker} in operator {this}, total of {resultHooksToInvoke.Count} hooks to invoke.");
		}

		/// <summary>
		/// This method blocks until the last state change has been fully performed.
		/// Returns immediately if not implemented.
		/// </summary>
		public void WaitForStateChanged()
		{
			lock (_stateChangeLock) { }
		}

		/// <summary>
		/// This method assures that <see cref="Workers"/> is initialised (with <see cref="InitialiseWorkers"/>)
		/// and checks if all required parameters are set. 
		/// </summary>
		protected virtual void PrepareWorkers()
		{
			// TODO: check if all required parameter are set
			// TODO uncomment this code and add more parameter checks
			//if (Trainer == null) throw new InvalidOperationException($"{nameof(Trainer)} cannot be null.");
			//if (Trainer.TrainingDataIterator == null) throw new InvalidOperationException($"{nameof(Trainer.TrainingDataIterator)} cannot be null.");
			//if (NetworkMerger == null) throw new InvalidOperationException($"{nameof(NetworkMerger)} cannot be null.");

			if (Workers == null)
			{
				Workers = InitialiseWorkers();
			}
		}

		/// <summary>
		///     This method creates the <see cref="IWorker" />s. It will be called before the first start of the operator.
		///     The <see cref="IWorker" />s are usually created via <see cref="CreateWorker" />.
		/// </summary>
		/// <returns>An <see cref="IEnumerable{T}" /> with the required amount of <see cref="IWorker" />s.</returns>
		protected virtual IEnumerable<IWorker> InitialiseWorkers()
		{
			IWorker[] workers = new IWorker[WorkerCount];
			IDictionary<IWorker, int> workerIndicesByWorkers = new Dictionary<IWorker, int>();

			for (int i = 0; i < workers.Length; i++)
			{
				workers[i] = CreateWorker();
				workers[i].LocalTrainingDataIterator = Trainer?.TrainingDataIterator?.ShallowCopy(); // TODO remove null conditional access, its only to pass operator/worker tests without trainer
				workers[i].LocalOptimiser = (IOptimiser) Trainer?.Optimiser?.DeepCopy();
				workerIndicesByWorkers.Add(workers[i], i);
			}

			WorkerIndicesByWorkers = new ReadOnlyDictionary<IWorker, int>(workerIndicesByWorkers);

			return workers;
		}

		/// <summary>
		/// Start all workers with <see cref="StartWorker"/>.
		/// </summary>
		protected virtual void StartWorkers()
		{
			foreach (IWorker worker in Workers)
			{
				StartWorker(worker);
			}
		}

		/// <summary>
		///		Start all workers once (for one iteration) with <see cref="RunWorkerOnce"/>. 
		/// </summary>
		protected virtual void StartWorkersOnce()
		{
			foreach (IWorker worker in Workers)
			{
				RunWorkerOnce(worker);
			}
		}

		#region StateControl

		public virtual void StartOnce()
		{
			if ((State == ExecutionState.None) || (State == ExecutionState.Stopped))
			{
				new BlockingLockingThread(_stateChangeLock, () =>
				{
					PrepareWorkers();

					StartWorkersOnce();

					State = ExecutionState.Running;
				}).Start();
			}
			else
			{
				ThrowBadState("started");
			}
		}

		/// <summary>
		///     Start this operator in a separate thread (return immediately).
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is running or paused.</exception>
		public void Start()
		{
			if ((State == ExecutionState.None) || (State == ExecutionState.Stopped))
			{
				new BlockingLockingThread(_stateChangeLock, () =>
				{
					PrepareWorkers();

					StartWorkers();

					State = ExecutionState.Running;
				}).Start();
			}
			else
			{
				ThrowBadState("started");
			}
		}

		/// <summary>
		///     Signal this operator to stop as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not running.</exception>
		public void SignalPause()
		{
			if (State == ExecutionState.Running)
			{
				new BlockingLockingThread(_stateChangeLock, () =>
				{
					foreach (IWorker worker in Workers) { PauseWorker(worker); }

					State = ExecutionState.Paused;
				}).Start();
			}
			else
			{
				ThrowBadState("paused");
			}
		}

		/// <summary>
		///     Signal this operator to resume as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not paused.</exception>
		public void SignalResume()
		{
			if (State == ExecutionState.Paused)
			{
				new BlockingLockingThread(_stateChangeLock, () =>
				 {
					 foreach (IWorker worker in Workers) { ResumeWorker(worker); }

					 State = ExecutionState.Running;
				 }).Start();
			}
			else
			{
				ThrowBadState("resumed");
			}
		}

		/// <summary>
		///     Signal this operator to stop as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is already stopped.</exception>
		public void SignalStop()
		{
			if (State != ExecutionState.Stopped)
			{
				new BlockingLockingThread(_stateChangeLock, () =>
				 {
					 foreach (IWorker worker in Workers)
					 {
						 PauseWorker(worker);
						 StopWorker(worker);
					 }

					 State = ExecutionState.Stopped;
				 }).Start();
			}
			else
			{
				ThrowBadState("stopped");
			}
		}

		/// <summary>
		/// </summary>
		/// <param name="currentState"></param>
		/// <exception cref="InvalidOperationException"></exception>
		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The operator cannot be {currentState} because the state is: {State}!");
		}

		#endregion

		/// <summary>
		///		Populate a registry using a certain worker's local values.
		/// </summary>
		/// <param name="registry">The registry to populate.</param>
		/// <param name="worker">The worker to fetch local values from.</param>
		public void PopulateWorkerRegistry(IRegistry registry, IWorker worker)
		{
			registry.Clear();

			UpdateRegistry(registry, worker.LocalNetwork, worker.LocalOptimiser, worker.LocalTrainingDataIterator, worker.LocalEpochNumber, worker.LocalIterationNumber);
		}

		/// <summary>
		/// Update a given registry with certain local values (typically for workers convenience).
		/// </summary>
		/// <param name="registry">The registry to update.</param>
		/// <param name="localNetwork">The local network.</param>
		/// <param name="localOptimiser">The local optimiser.</param>
		/// <param name="localIterator">The local data iterator.</param>
		/// <param name="localEpochNumber">The local epoch number.</param>
		/// <param name="localIterationNumber">The local iteration number.</param>
		protected void UpdateRegistry(IRegistry registry, INetwork localNetwork, IOptimiser localOptimiser, IDataIterator localIterator,
			int localEpochNumber, int localIterationNumber)
		{
			if (registry == null) throw new ArgumentNullException(nameof(registry));
			if (localNetwork == null) throw new ArgumentNullException(nameof(localNetwork));
			if (localOptimiser == null) throw new ArgumentNullException(nameof(localOptimiser));
			if (localIterator == null) throw new ArgumentNullException(nameof(localIterator));

			registry["network"] = localNetwork.Registry;
			registry["optimiser"] = localOptimiser.Registry;
			registry["iterator"] = localIterator.Registry;
			registry["epoch"] = localEpochNumber;
			registry["iteration"] = localIterationNumber;
		}

		public abstract void PushProgress(IWorker worker);
		public abstract void PullProgress(IWorker worker);

		#region AbstractWorkerMethods

		/// <summary>
		///     This method creates an <see cref="IWorker" />.
		/// </summary>
		/// <returns>The newly created <see cref="IWorker" />.</returns>
		protected abstract IWorker CreateWorker();

		/// <summary>
		///     This method starts a worker.
		/// </summary>
		/// <param name="worker">The worker that will be started.</param>
		protected abstract void StartWorker(IWorker worker);

		/// <summary>
		///     This method starts a worker for a single iteration.
		/// </summary>
		/// <param name="worker">The worker that will be started.</param>
		protected abstract void RunWorkerOnce(IWorker worker);

		/// <summary>
		///     This method pauses a worker. It will also be
		///     called if the worker is stopped.
		/// </summary>
		/// <param name="worker">The worker that will be paused.</param>
		protected abstract void PauseWorker(IWorker worker);

		/// <summary>
		///     This method resumes a worker from it's paused state.
		/// </summary>
		/// <param name="worker">The worker that will be resumed.</param>
		protected abstract void ResumeWorker(IWorker worker);

		/// <summary>
		///     This method stops a worker. All resources should
		///     be freed.
		/// </summary>
		/// <param name="worker">The worker that will be paused and stopped.</param>
		protected abstract void StopWorker(IWorker worker);

		#endregion AbstractWorkerMethods
	}
}