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
		public OperatorState State { get; protected set; } = OperatorState.None;
		public IComputationHandler Handler { get; }
		public ITrainer Trainer { get; }
		public INetwork Network { get; }
		public int WorkerCount { get; }

		protected IEnumerable<IWorker> Workers;

		protected IList<IHook> Hooks;

		protected BaseOperator(SigmaEnvironment sigma, IComputationHandler handler, ITrainer trainer, INetwork network, int workerCount)
		{
			Sigma = sigma;
			Handler = handler;
			Trainer = trainer;
			Network = network;
			WorkerCount = workerCount;

			Hooks = new List<IHook>();

			Workers = InitialiseWorkers();
		}

		protected virtual IEnumerable<IWorker> InitialiseWorkers()
		{
			IWorker[] workers = new IWorker[WorkerCount];

			for (int i = 0; i < workers.Length; i++)
			{
				workers[i] = CreateWorker(this);
			}

			return workers;
		}

		public abstract IWorker CreateWorker(IOperator @operator);

		/// <summary>
		///		This method starts a worker. 
		/// </summary>
		/// <param name="worker">The worker that will be started.</param>
		public abstract void StartWorker(IWorker worker);

		/// <summary>
		///		This method pauses a worker. It will also be
		///		called if the worker is stopped.
		/// </summary>
		/// <param name="worker">The worker that will be paused.</param>
		public abstract void PauseWorker(IWorker worker);

		/// <summary>
		///		This method resumes a worker from it's paused state.
		/// </summary>
		/// <param name="worker">The worker that will be resumed.</param>
		public abstract void ResumeWorker(IWorker worker);

		/// <summary>
		///		This method stops a worker. All resources should
		/// be freed. 
		/// </summary>
		/// <param name="worker">The worker that will be paused and stopped.</param>
		public abstract void StopWorker(IWorker worker);


		public void AttachHook(IHook hook)
		{
			Hooks.Add(hook);
		}

		public void DetachHook(IHook hook)
		{
			Hooks.Remove(hook);
		}

		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The operator cannot be {currentState} because the state is: {State}!");
		}

		public void Start()
		{
			if (State == OperatorState.None || State == OperatorState.Stopped)
			{
				foreach (IWorker worker in Workers)
				{
					StartWorker(worker);
				}

				State = OperatorState.Running;
			}
			else
			{
				ThrowBadState("started");
			}
		}

		public void SignalPause()
		{
			if (State == OperatorState.Running)
			{
				foreach (IWorker worker in Workers)
				{
					PauseWorker(worker);
				}

				State = OperatorState.Paused;
			}
			else
			{
				ThrowBadState("paused");
			}
		}

		public void SignalResume()
		{
			if (State == OperatorState.Paused)
			{
				foreach (IWorker worker in Workers)
				{
					ResumeWorker(worker);
				}

				State = OperatorState.Running;
			}
			else
			{
				ThrowBadState("resumed");
			}
		}

		public void SignalStop()
		{
			if (State != OperatorState.Stopped)
			{
				foreach (IWorker worker in Workers)
				{
					PauseWorker(worker);
					StopWorker(worker);
				}

				State = OperatorState.Stopped;
			}
			else
			{
				ThrowBadState("stopped");
			}
		}

	}
}