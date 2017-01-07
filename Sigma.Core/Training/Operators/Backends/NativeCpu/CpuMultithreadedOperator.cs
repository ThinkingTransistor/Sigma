/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Backends.NativeCpu.Workers;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu
{
	/// <summary>
	///     This class is just an alias for the <see cref="CpuMultithreadedOperator" />.
	///     On some cases it can be useful to have separate classes.
	///     It runs only on a single thread.
	/// </summary>
	public class CpuSinglethreadedOperator : CpuMultithreadedOperator
	{
		/// <summary>
		///     Create a new <see cref="CpuSinglethreadedOperator" /> without specifying the <see cref="IComputationHandler" />.
		///     The <see cref="IComputationHandler" /> will be automatically set by the <see cref="ITrainer" />.
		///     The <see cref="ThreadPriority" /> will receive its default value (<see cref="ThreadPriority.Highest" />).
		/// </summary>
		public CpuSinglethreadedOperator() : this(null, ThreadPriority.Highest)
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
		protected CpuSinglethreadedOperator(IComputationHandler handler, ThreadPriority priority) : base(handler, 1, priority)
		{
		}
	}

	/// <summary>
	///     This is a multithreaded CPU-based operator. It will execute all
	///     operations on the CPU. The tasks will be executed concurrently by the
	///     number of threads specified.
	/// </summary>
	public class CpuMultithreadedOperator : BaseOperator
	{
		/// <summary>
		///     The <see cref="ThreadPriority" /> with which newly created Threads
		///     will be started. This is <see cref="ThreadPriority.Highest" /> per default.
		/// </summary>
		public ThreadPriority WorkerPriority { get; set; }

		/// <summary>
		/// The current iteration number, with all networks. 
		/// </summary>
		private readonly IDictionary<int, INetwork[]> _pushedNetworks;

		/// <summary>
		///     Create a new <see cref="CpuMultithreadedOperator" /> without specifying the <see cref="IComputationHandler" />.
		///     The <see cref="IComputationHandler" /> will be automatically set by the <see cref="ITrainer" />.
		///     The <see cref="ThreadPriority" /> will receive its default value (<see cref="ThreadPriority.Highest" />).
		/// </summary>
		/// <param name="workerCount">
		///     The number of <see cref="IWorker" />s (threads) used in this <see cref="IOperator" /> in
		///     parallel.
		/// </param>
		public CpuMultithreadedOperator(int workerCount) : this(null, workerCount, ThreadPriority.Highest)
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
			_pushedNetworks = new Dictionary<int, INetwork[]>();
		}

		public override void ReportProgress(IWorker worker)
		{
			PushNetwork(worker);
		}

		protected virtual void PushNetwork(IWorker worker)
		{
			lock (_pushedNetworks)
			{
				INetwork[] networks = _pushedNetworks.TryGetValue(worker.LocalIterationNumber, () => new INetwork[WorkerCount]);
				if (!networks.AddToNextNull(worker.LocalNetwork))
				{
					throw new InvalidOperationException("Too many workers trying to push their network");
				}
			}

			Log.Info($"Worker {worker.GetType()} pushed his network for the iteration {worker.LocalIterationNumber}");
		}

		/// <summary>
		///     This implementation of the <see cref="BaseOperator" /> creates a new <see cref="BaseCpuWorker" />.
		///     <see cref="IComputationHandler" /> and <see cref="WorkerPriority" /> are assigned correctly.
		/// </summary>
		/// <returns>Return a newly created <see cref="BaseCpuWorker" />.</returns>
		protected override IWorker CreateWorker()
		{
			return new CpuWorker(this, Handler, WorkerPriority);
		}

		protected override void StartWorker(IWorker worker)
		{
			worker.Start();
		}

		protected override void RunWorkerOnce(IWorker worker)
		{
			worker.RunOnce();
		}

		protected override void PauseWorker(IWorker worker)
		{
			worker.SignalPause();
		}

		protected override void ResumeWorker(IWorker worker)
		{
			worker.SignalResume();
		}

		protected override void StopWorker(IWorker worker)
		{
			worker.SignalStop();
		}
	}
}