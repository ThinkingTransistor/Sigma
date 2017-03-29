/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Training.Operators.Backends.NativeCpu.Workers;
using Sigma.Core.Training.Operators.Workers;
using System.Threading;
using Sigma.Core.Persistence.Selectors;
using Sigma.Core.Persistence.Selectors.Operator;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu
{
	/// <summary>
	///     This class is just an alias for the <see cref="CpuMultithreadedOperator" />.
	///     On some cases it can be useful to have separate classes.
	///     It runs only on a single thread.
	/// </summary>
	[Serializable]
	public class CpuSinglethreadedOperator : CpuMultithreadedOperator
	{
		/// <summary>
		///     Create a new <see cref="CpuSinglethreadedOperator" /> using the default <see cref="IComputationHandler" /> (<see cref="CpuFloat32Handler"/>).
		///     The <see cref="ThreadPriority" /> will receive its default value (<see cref="ThreadPriority.Highest" />).
		/// </summary>
		public CpuSinglethreadedOperator() : this(new CpuFloat32Handler(), ThreadPriority.Highest)
		{
		}

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> with a specified <see cref="IComputationHandler" /> and
		///     <see cref="ThreadPriority" />.
		///     The <see cref="IComputationHandler" /> will <c>not</c> be modified by the <see cref="ITrainer" />.
		/// </summary>
		/// <param name="handler">
		///     The <see cref="IComputationHandler" /> that will be assigned to the
		///     <see cref="IComputationHandler" />
		/// </param>
		/// <param name="priority">The <see cref="ThreadPriority" /> with which the newly created Thread will be started. </param>
		public CpuSinglethreadedOperator(IComputationHandler handler, ThreadPriority priority = ThreadPriority.Highest) : base(handler, 1, priority)
		{
		}

		/// <summary>
		/// Create an instance of this operator with the same parameters.
		/// Used for shallow-copying state to another operator (e.g. for persistence / selection).
		/// </summary>
		/// <returns></returns>
		protected override BaseOperator CreateDuplicateInstance()
		{
			return new CpuSinglethreadedOperator(Handler, WorkerPriority);
		}

		/// <summary>
		/// Get an operator selector for this operator.
		/// </summary>
		/// <returns>The selector for this operator.</returns>
		public override IOperatorSelector<IOperator> Select()
		{
			return new CpuSinglethreadedOperatorSelector(this);
		}
	}

	/// <summary>
	///     This is a multithreaded CPU-based operator. It will execute all
	///     operations on the CPU. The tasks will be executed concurrently by the
	///     number of threads specified.
	/// </summary>
	[Serializable]
	public class CpuMultithreadedOperator : BaseOperator
	{
		/// <summary>
		///     The <see cref="ThreadPriority" /> with which newly created Threads
		///     will be started. This is <see cref="ThreadPriority.Highest" /> per default.
		/// </summary>
		public ThreadPriority WorkerPriority { get; }

		/// <summary>
		///     Create a new <see cref="CpuMultithreadedOperator" /> using the default <see cref="IComputationHandler" /> (<see cref="CpuFloat32Handler"/>).
		///     The <see cref="ThreadPriority" /> will receive its default value (<see cref="ThreadPriority.Highest" />).
		/// </summary>
		/// <param name="workerCount">
		///     The number of <see cref="IWorker" />s (threads) used in this <see cref="IOperator" /> in
		///     parallel.
		/// </param>
		public CpuMultithreadedOperator(int workerCount) : this(new CpuFloat32Handler(), workerCount, ThreadPriority.Highest)
		{
		}

		/// <summary>
		///     Create a new <see cref="BaseOperator" /> with a specified <see cref="IComputationHandler" /> and
		///     <see cref="ThreadPriority" />.
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
		/// <param name="priority">The <see cref="ThreadPriority" /> with which newly created Threads will be started.</param>
		public CpuMultithreadedOperator(IComputationHandler handler, int workerCount, ThreadPriority priority)
			: base(handler, workerCount)
		{
			WorkerPriority = priority;
		}

		/// <summary>
		///     This implementation of the <see cref="BaseOperator" /> creates a new <see cref="CpuWorker" />.
		///     <see cref="IComputationHandler" /> and <see cref="WorkerPriority" /> are assigned correctly.
		/// </summary>
		/// <returns>Return a newly created <see cref="CpuWorker" />.</returns>
		protected override IWorker CreateWorker()
		{
			return new CpuWorker(this, Handler, WorkerPriority);
		}

		/// <inheritdoc />
		protected override void StartWorker(IWorker worker)
		{
			Logger.Debug($"Starting worker {worker} in operator {this}...");

			worker.Start();
		}

		/// <inheritdoc />
		protected override void RunWorkerOnce(IWorker worker)
		{
			Logger.Debug($"Running worker {worker} once in operator {this}...");

			worker.RunOnce();
		}

		/// <inheritdoc />
		protected override void PauseWorker(IWorker worker)
		{
			Logger.Debug($"Signalling pause to worker {worker} in operator {this}...");

			worker.SignalPause();
		}

		/// <inheritdoc />
		protected override void ResumeWorker(IWorker worker)
		{
			Logger.Debug($"Signalling resume to worker {worker} in operator {this}...");

			worker.SignalResume();
		}

		/// <inheritdoc />
		protected override void StopWorker(IWorker worker)
		{
			Logger.Debug($"Stopping worker {worker} in operator {this}...");

			worker.SignalStop();
		}

		/// <summary>
		/// Create an instance of this operator with the same parameters.
		/// Used for shallow-copying state to another operator (e.g. for persistence / selection).
		/// </summary>
		/// <returns></returns>
		protected override BaseOperator CreateDuplicateInstance()
		{
			return new CpuMultithreadedOperator(Handler, WorkerCount, WorkerPriority);
		}

		/// <summary>
		/// Get an operator selector for this operator.
		/// </summary>
		/// <returns>The selector for this operator.</returns>
		public override IOperatorSelector<IOperator> Select()
		{
			return new CpuMultithreadedOperatorSelector(this);
		}
	}
}