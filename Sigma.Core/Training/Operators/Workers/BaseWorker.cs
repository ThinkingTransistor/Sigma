/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Workers
{
	public abstract class BaseWorker : IWorker
	{
		/// <summary>
		/// The priority with which the <see cref="WorkerThread"/> will be started. 
		/// <see cref="ThreadPriority.Highest"/> per default.
		/// </summary>
		public ThreadPriority ThreadPriority { get; set; }

		public IOperator Operator { get; }
		public ExecutionState State { get; protected set; } = ExecutionState.None;
		public IComputationHandler Handler { get; }
		public INetwork LocalNetwork { get; set; }
		public IDataIterator LocalTrainingDataIterator { get; set; }
		public IOptimiser LocalOptimiser { get; set; }

		public int LocalEpochNumber { get; protected set; }
		public int LocalIterationNumber { get; protected set; }

		/// <summary>
		/// The thread that executes the update (see <see cref="Update"/>). 
		/// </summary>
		protected Thread WorkerThread { get; set; }

		private readonly ISet<IHook> _bufferHooksToInvoke;
		private readonly ISet<IHook> _bufferHooksInBackgroundToInvoke;
		private readonly ISet<string> _bufferRegistryEntries;
		private readonly ISet<string> _bufferResolvedRegistryEntries;
		private readonly IRegistry _bufferRegistry;
		private readonly IRegistryResolver _bufferRegistryResolver;
		private readonly object _stateLock;

		/// <summary>
		/// The event that locks the <see cref="WorkerThread"/> until the execution resumes (see <see cref="SignalResume"/>). 
		/// </summary>
		private readonly ManualResetEvent _waitForResume;

		/// <summary>
		///		The time scale countdowns per passive hook (passive hooks are managed by the operator).
		/// </summary>
		protected readonly IDictionary<IHook, ITimeStep> LocalActiveHookTimeSteps;

		protected BaseWorker(IOperator @operator, ThreadPriority priority = ThreadPriority.Highest) : this(@operator, @operator.Handler, priority)
		{
		}

		protected BaseWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority = ThreadPriority.Highest)
		{
			Operator = @operator;
			Handler = handler;
			LocalActiveHookTimeSteps = new Dictionary<IHook, ITimeStep>();
			ThreadPriority = priority;
			_bufferHooksToInvoke = new HashSet<IHook>();
			_bufferHooksInBackgroundToInvoke = new HashSet<IHook>();
			_bufferRegistryEntries = new HashSet<string>();
			_bufferResolvedRegistryEntries = new HashSet<string>();
			_bufferRegistry = new Registry();
			_bufferRegistryResolver = new RegistryResolver(_bufferRegistry);
			_stateLock = new object();
			_waitForResume = new ManualResetEvent(false);
		}

		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The {nameof(BaseWorker)} cannot be {currentState} because the state is: {State}!");
		}

		protected virtual Thread CreateThread(ThreadStart start)
		{
			return new Thread(start) { Priority = ThreadPriority };
		}

		public void Start()
		{
			if (State == ExecutionState.Stopped || State == ExecutionState.None)
			{
				lock (_stateLock)
				{
					Initialise();

					State = ExecutionState.Running;

					WorkerThread = CreateThread(Update);

					WorkerThread.Start();
				}
			}
			else if (State != ExecutionState.Running)
			{
				ThrowBadState("started");
			}
		}

		public void SignalPause()
		{
			if (State == ExecutionState.Running)
			{
				lock (_stateLock)
				{
					OnPause();

					State = ExecutionState.Paused;
				}
			}
			else if (State != ExecutionState.Paused)
			{
				ThrowBadState("paused");
			}
		}

		public void SignalResume()
		{
			if (State == ExecutionState.Paused)
			{
				lock (_stateLock)
				{
					OnResume();

					State = ExecutionState.Running;

					_waitForResume.Set();
				}
			}
			else if (State != ExecutionState.Running)
			{
				ThrowBadState("resumed");
			}
		}

		public void SignalStop()
		{
			// if the worker is not already stopped
			if (State != ExecutionState.None && State != ExecutionState.Stopped)
			{
				lock (_stateLock)
				{
					// if its not paused, signal it to pause
					if (State != ExecutionState.Paused)
					{
						SignalPause();
					}

					OnStop();

					State = ExecutionState.Stopped;
					_waitForResume.Set();
				}
			}
		}

		public void RunOnce()
		{
			if (State != ExecutionState.Running)
			{
				lock (_stateLock)
				{
					if (State == ExecutionState.None || State == ExecutionState.Stopped)
					{
						Initialise();
					}
					else //Paused
					{
						OnResume();
					}

					new ThreadUtils.BlockingThread(reset =>
					{
						DoWork();
						reset.Set();
					}).Start();

					OnStop();
				}
			}
			else
			{
				ThrowBadState("started once");
			}
		}

		/// <summary>
		/// This method will be called every time the worker will start from a full stop.
		/// </summary>
		protected abstract void Initialise();

		/// <summary>
		/// This method gets updated as long as this worker is not paused or stopped. Maybe only once. 
		/// </summary>
		protected abstract void DoWork();

		/// <summary>
		/// This method gets called when the worker is about to pause. It will also be called when the worker
		/// is stopped.
		/// </summary>
		protected abstract void OnPause();

		/// <summary>
		/// This method gets called when the worker resumes from its paused state. 
		/// </summary>
		protected abstract void OnResume();

		/// <summary>
		/// This method gets called when the worker comes to a halt.
		/// </summary>
		protected abstract void OnStop();

		/// <summary>
		/// Update all the tasks. This method runs as long as it's not <see cref="ExecutionState.Stopped"/>.
		/// </summary>
		private void Update()
		{
			while (State != ExecutionState.Stopped)
			{
				while (State == ExecutionState.Running)
				{
					DoWork();
				}

				if (State == ExecutionState.Paused)
				{
					_waitForResume.WaitOne();
					_waitForResume.Reset();
				}
			}

			_waitForResume.Reset();
		}

		public void InvokeTimeScaleEvent(TimeScale timeScale)
		{
			var activeHooks = Operator.Trainer.ActiveHooks;

			Operator.EjectTimeScaleEvent(timeScale, activeHooks, LocalActiveHookTimeSteps, _bufferHooksToInvoke);
			MarkDeadHooks(activeHooks, LocalActiveHookTimeSteps);

			Operator.PopulateWorkerRegistry(_bufferRegistry, this);

			HookUtils.FetchBackgroundHooks(_bufferHooksToInvoke, _bufferHooksInBackgroundToInvoke);

			foreach (IHook hook in _bufferHooksToInvoke)
			{
				if (!hook.InvokeInBackground)
				{
					hook.Invoke(_bufferRegistry, _bufferRegistryResolver);
				}
			}

			if (_bufferHooksInBackgroundToInvoke.Count > 0)
			{
				Operator.DispatchBackgroundHooks(_bufferHooksInBackgroundToInvoke, _bufferRegistry, _bufferRegistryEntries, _bufferResolvedRegistryEntries);
			}
		}

		protected void MarkDeadHooks(IEnumerable<IActiveHook> hooks, IDictionary<IHook, ITimeStep> localTimeSteps)
		{
			foreach (IHook hook in _bufferHooksToInvoke)
			{
				IActiveHook asActiveHook = hook as IActiveHook;

				if (asActiveHook == null)
				{
					throw new InvalidOperationException($"Internal error: Buffered hooks to invoke in worker may only contain active hooks but hook {hook} could not be cast accordingly.");
				}

				if (LocalActiveHookTimeSteps[hook].LocalLiveTime == 0)
				{
					Operator.MarkHookDead(asActiveHook, this);
				}
			}
		}
	}
}