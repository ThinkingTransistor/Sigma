/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Threading;

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

		private readonly List<IHook> _bufferHooksToInvoke;
		private readonly IList<IHook> _bufferHooksToInvokeInBackground;
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
		///		The time scale countdowns per local hook (global hooks are managed by the operator).
		/// </summary>
		protected readonly IDictionary<IHook, ITimeStep> LocalLocalHookTimeSteps;

		protected BaseWorker(IOperator @operator, ThreadPriority priority = ThreadPriority.Highest) : this(@operator, @operator.Handler, priority)
		{
		}

		protected BaseWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority = ThreadPriority.Highest)
		{
			Operator = @operator;
			Handler = handler;
			LocalLocalHookTimeSteps = new Dictionary<IHook, ITimeStep>();
			ThreadPriority = priority;
			_bufferHooksToInvoke = new List<IHook>();
			_bufferHooksToInvokeInBackground = new List<IHook>();
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

					InvokeTimeScaleEvent(TimeScale.Start);
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

					InvokeTimeScaleEvent(TimeScale.Pause);
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

					InvokeTimeScaleEvent(TimeScale.Resume);
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

					InvokeTimeScaleEvent(TimeScale.Stop);
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
			Operator.EjectTimeScaleEvent(timeScale, Operator.AttachedLocalHooksByTimeScale, LocalLocalHookTimeSteps, _bufferHooksToInvoke);
			MarkDeadHooks(Operator.AttachedLocalHooks, LocalLocalHookTimeSteps);

			Operator.PopulateWorkerRegistry(_bufferRegistry, this);

			ArrayUtils.SortListInPlaceIndexed(_bufferHooksToInvoke, Operator.GetLocalHookInvocationIndex);
			HookUtils.FetchOrderedBackgroundHooks(_bufferHooksToInvoke, _bufferHooksToInvokeInBackground);

			foreach (IHook hook in _bufferHooksToInvoke)
			{
				if (!hook.InvokeInBackground)
				{
					hook.Operator = Operator;
					hook.Invoke(_bufferRegistry, _bufferRegistryResolver);
				}
			}

			if (_bufferHooksToInvokeInBackground.Count > 0)
			{
				Operator.DispatchBackgroundHookInvocation(_bufferHooksToInvokeInBackground, _bufferRegistry, _bufferRegistryEntries, _bufferResolvedRegistryEntries);
			}
		}

		private void MarkDeadHooks(IEnumerable<IHook> hooks, IDictionary<IHook, ITimeStep> localTimeSteps)
		{
			foreach (IHook hook in _bufferHooksToInvoke)
			{
				if (LocalLocalHookTimeSteps[hook].LocalLiveTime == 0)
				{
					Operator.MarkHookDead(hook, this);
				}
			}
		}
	}
}