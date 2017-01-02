/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Operators.Workers;
// ReSharper disable VirtualMemberCallInConstructor

namespace Sigma.Core.Training.Operators
{
	public abstract class BaseOperator : IOperator
	{
		public SigmaEnvironment Sigma { get; set; }
		public ExecutionState State { get; protected set; } = ExecutionState.None;
		public IComputationHandler Handler { get; }
		public ITrainer Trainer { get; set; }
		public INetwork Network { get; set; }
		public int WorkerCount { get; }

		protected IEnumerable<IWorker> Workers;

		protected IList<IHook> Hooks;

		protected BaseOperator(SigmaEnvironment sigma, IComputationHandler handler, int workerCount)
		{
			Sigma = sigma;
			Handler = handler;
			WorkerCount = workerCount;

			Hooks = new List<IHook>();
		}

		protected virtual IEnumerable<IWorker> InitialiseWorkers()
		{
			IWorker[] workers = new IWorker[WorkerCount];

			for (int i = 0; i < workers.Length; i++)
			{
				workers[i] = CreateWorker();
			}

			return workers;
		}

		protected abstract IWorker CreateWorker();

		/// <summary>
		///		This method starts a worker. 
		/// </summary>
		/// <param name="worker">The worker that will be started.</param>
		protected abstract void StartWorker(IWorker worker);

		/// <summary>
		///		This method pauses a worker. It will also be
		///		called if the worker is stopped.
		/// </summary>
		/// <param name="worker">The worker that will be paused.</param>
		protected abstract void PauseWorker(IWorker worker);

		/// <summary>
		///		This method resumes a worker from it's paused state.
		/// </summary>
		/// <param name="worker">The worker that will be resumed.</param>
		protected abstract void ResumeWorker(IWorker worker);

		/// <summary>
		///		This method stops a worker. All resources should
		/// be freed. 
		/// </summary>
		/// <param name="worker">The worker that will be paused and stopped.</param>
		protected abstract void StopWorker(IWorker worker);


		public void AttachHook(IHook hook)
		{
			Hooks.Add(hook);
		}

		public void DetachHook(IHook hook)
		{
			Hooks.Remove(hook);
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="currentState"></param>
		/// <exception cref="InvalidOperationException"></exception>
		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The operator cannot be {currentState} because the state is: {State}!");
		}

		/// <summary>
		/// Start this operator in a separate thread (return immediately). 
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is running or paused.</exception>
		public void Start()
		{
			if (State == ExecutionState.None || State == ExecutionState.Stopped)
			{
				// initialise the workers when they are required the first time
				if (Workers == null)
				{
					Workers = InitialiseWorkers();
				}

				foreach (IWorker worker in Workers)
				{
					StartWorker(worker);
				}

				State = ExecutionState.Running;
			}
			else
			{
				ThrowBadState("started");
			}
		}

		/// <summary>
		/// Signal this operator to stop as soon as possible. 
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not running.</exception>
		public void SignalPause()
		{
			if (State == ExecutionState.Running)
			{
				foreach (IWorker worker in Workers)
				{
					PauseWorker(worker);
				}

				State = ExecutionState.Paused;
			}
			else
			{
				ThrowBadState("paused");
			}
		}

		/// <summary>
		/// Signal this operator to resume as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not paused.</exception>
		public void SignalResume()
		{
			if (State == ExecutionState.Paused)
			{
				foreach (IWorker worker in Workers)
				{
					ResumeWorker(worker);
				}

				State = ExecutionState.Running;
			}
			else
			{
				ThrowBadState("resumed");
			}
		}

		/// <summary>
		/// Signal this operator to stop as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is already stopped.</exception>
		public void SignalStop()
		{
			if (State != ExecutionState.Stopped)
			{
				foreach (IWorker worker in Workers)
				{
					PauseWorker(worker);
					StopWorker(worker);
				}

				State = ExecutionState.Stopped;
			}
			else
			{
				ThrowBadState("stopped");
			}
		}

	}
}