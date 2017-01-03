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

namespace Sigma.Core.Training.Operators
{
	public abstract class BaseOperator : IOperator
	{
		/// <summary>
		///     All <see cref="IHook" />s that are attached to this <see cref="IOperator" />.
		/// </summary>
		protected ICollection<IHook> Hooks;

		/// <summary>
		///     All the <see cref="IWorker" />s managed by this operator.
		/// </summary>
		protected IEnumerable<IWorker> Workers;

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> without specifying the <see cref="IComputationHandler" />.
		///     The <see cref="IComputationHandler" /> will be automatically set by the <see cref="ITrainer" />.
		/// </summary>
		/// <param name="workerCount">
		///     The number of <see cref="IWorker" />s (threads) used in this <see cref="IOperator" /> in
		///     parallel.
		/// </param>
		protected BaseOperator(int workerCount)
		{
			Hooks = new List<IHook>();
			WorkerCount = workerCount;
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
		protected BaseOperator(IComputationHandler handler, int workerCount) : this(workerCount)
		{
			Handler = handler;
		}

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
		///     The number of <see cref="IWorker" />s (threads) used in this
		///     <see cref="IOperator" /> in parallel.
		/// </summary>
		public int WorkerCount { get; }

		public void AttachHook(IHook hook)
		{
			Hooks.Add(hook);
		}

		public void DetachHook(IHook hook)
		{
			Hooks.Remove(hook);
		}

		/// <summary>
		///     Start this operator in a separate thread (return immediately).
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is running or paused.</exception>
		public void Start()
		{
			if ((State == ExecutionState.None) || (State == ExecutionState.Stopped))
			{
				// TODO: check if all required parameter are set

				// initialise the workers when they are required the first time
				if (Workers == null) { Workers = InitialiseWorkers(); }

				foreach (IWorker worker in Workers) { StartWorker(worker); }

				State = ExecutionState.Running;
			}
			else
			{ ThrowBadState("started"); }
		}

		/// <summary>
		///     Signal this operator to stop as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not running.</exception>
		public void SignalPause()
		{
			if (State == ExecutionState.Running)
			{
				foreach (IWorker worker in Workers) { PauseWorker(worker); }

				State = ExecutionState.Paused;
			}
			else
			{ ThrowBadState("paused"); }
		}

		/// <summary>
		///     Signal this operator to resume as soon as possible.
		/// </summary>
		/// <exception cref="InvalidOperationException">If the operator is not paused.</exception>
		public void SignalResume()
		{
			if (State == ExecutionState.Paused)
			{
				foreach (IWorker worker in Workers) { ResumeWorker(worker); }

				State = ExecutionState.Running;
			}
			else
			{ ThrowBadState("resumed"); }
		}

		/// <summary>
		///     Signal this operator to stop as soon as possible.
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
			{ ThrowBadState("stopped"); }
		}

		/// <summary>
		///     This method creates the <see cref="IWorker" />s. It will be called before the first start of the operator.
		///     The <see cref="IWorker" />s are usually created via <see cref="CreateWorker" />.
		/// </summary>
		/// <returns>An <see cref="IEnumerable{T}" /> with the required amount of <see cref="IWorker" />s.</returns>
		protected virtual IEnumerable<IWorker> InitialiseWorkers()
		{
			IWorker[] workers = new IWorker[WorkerCount];

			for (int i = 0; i < workers.Length; i++) { workers[i] = CreateWorker(); }

			return workers;
		}

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

		/// <summary>
		/// </summary>
		/// <param name="currentState"></param>
		/// <exception cref="InvalidOperationException"></exception>
		private void ThrowBadState(string currentState)
		{
			throw new InvalidOperationException($"The operator cannot be {currentState} because the state is: {State}!");
		}
	}
}