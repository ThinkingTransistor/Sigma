/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Threading;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public abstract class BaseCpuWorker : BaseWorker
	{
		/// <summary>
		/// The thread that executes the update (see <see cref="Update"/>). 
		/// </summary>
		protected Thread WorkerThread { get; set; }

		/// <summary>
		/// The priority with which the <see cref="WorkerThread"/> will be started. 
		/// <see cref="ThreadPriority.Highest"/> per default.
		/// </summary>
		public ThreadPriority ThreadPriority { get; set; }

		private readonly object _stateLock;

		/// <summary>
		/// The event that locks the <see cref="WorkerThread"/> until the execution resumes (see <see cref="SignalResume"/>). 
		/// </summary>
		private readonly ManualResetEvent _waitForResume;

		protected BaseCpuWorker(IOperator @operator) : this(@operator, @operator.Handler)
		{
		}

		protected BaseCpuWorker(IOperator @operator, IComputationHandler handler) : this(@operator, handler, ThreadPriority.Highest)
		{

		}

		protected BaseCpuWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority) : base(@operator, handler)
		{
			_stateLock = new object();

			_waitForResume = new ManualResetEvent(false);
			ThreadPriority = priority;
		}

		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The {nameof(BaseCpuWorker)} cannot be {currentState} because the state is: {State}!");
		}

		protected virtual Thread CreateThread(ThreadStart start)
		{
			return new Thread(start) { Priority = ThreadPriority };
		}

		public override void Start()
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

		public override void SignalPause()
		{
			if (State == ExecutionState.Running)
			{
				lock (_stateLock)
				{
					Pause();

					State = ExecutionState.Paused;
				}
			}
			else if (State != ExecutionState.Paused)
			{
				ThrowBadState("paused");
			}
		}

		public override void SignalResume()
		{
			if (State == ExecutionState.Paused)
			{
				lock (_stateLock)
				{
					Resume();

					State = ExecutionState.Running;

					_waitForResume.Set();
				}
			}
			else if (State != ExecutionState.Running)
			{
				ThrowBadState("resumed");
			}
		}

		public override void SignalStop()
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

					Stop();

					State = ExecutionState.Stopped;
					_waitForResume.Set();
				}
			}
		}

		public override void RunTrainingIteration()
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
						Resume();
					}

					new ThreadUtils.BlockingThread(reset =>
					{
						DoWork();
						reset.Set();
					}).Start();

					Stop();
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
		/// This method gets updated as long as this worker is not paused or stopped. Maybe even once. 
		/// </summary>
		protected abstract void DoWork();

		/// <summary>
		/// This method gets called when the worker is about to pause. It will also be called when the worker
		/// is stopped.
		/// </summary>
		protected abstract void Pause();

		/// <summary>
		/// This method gets called when the worker resumes from its paused state. 
		/// </summary>
		protected abstract void Resume();

		/// <summary>
		/// This method gets called when the worker comes to a halt.
		/// </summary>
		protected abstract void Stop();

		/// <summary>
		/// Update all the tasks. This method runs as long as its not <see cref="ExecutionState.Stopped"/>.
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
	}
}