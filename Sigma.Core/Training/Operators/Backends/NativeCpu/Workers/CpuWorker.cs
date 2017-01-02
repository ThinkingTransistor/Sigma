/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Threading;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseWorker
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

		/// <summary>
		/// The event that locks the <see cref="WorkerThread"/> until the execution resumes (see <see cref="SignalResume"/>). 
		/// </summary>
		private readonly ManualResetEvent _waitForResume;

		public CpuWorker(IOperator @operator) : this(@operator, @operator.Handler)
		{
		}

		public CpuWorker(IOperator @operator, IComputationHandler handler) : this(@operator, handler, ThreadPriority.Highest)
		{

		}

		public CpuWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority) : base(@operator, handler)
		{
			_waitForResume = new ManualResetEvent(false);
			ThreadPriority = priority;
		}

		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The {nameof(CpuWorker)} cannot be {currentState} because the state is: {State}!");
		}

		public override void Start()
		{
			if (State == ExecutionState.Stopped || State == ExecutionState.None)
			{
				State = ExecutionState.Running;

				WorkerThread = new Thread(Update) { Priority = ThreadPriority };

				WorkerThread.Start();
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
				State = ExecutionState.Paused;
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
				State = ExecutionState.Running;

				_waitForResume.Set();
			}
			else if (State != ExecutionState.Running)
			{
				ThrowBadState("resumed");
			}
		}

		public override void SignalStop()
		{
			if (State == ExecutionState.Paused)
			{
				State = ExecutionState.Stopped;

				_waitForResume.Set();
			}
			else if (State != ExecutionState.Stopped)
			{
				throw new InvalidOperationException($"The {nameof(CpuWorker)} can only be stopped if it has been paused first!");
			}
		}

		/// <summary>
		/// Update all the tasks. This method runs as long as its not <see cref="ExecutionState.Stopped"/>.
		/// </summary>
		protected virtual void Update()
		{
			while (State != ExecutionState.Stopped)
			{
				while (State == ExecutionState.Running)
				{
					// TODO: work
					Thread.Sleep(100);
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