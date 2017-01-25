/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
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
		public ThreadPriority WorkerPriority { get; }

		/// <summary>
		/// The current epoch number, with all networks. 
		/// </summary>
		private readonly IDictionary<int, INetwork[]> _pushedNetworks;

		private readonly object _networkChangedLock;

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
			_pushedNetworks = new Dictionary<int, INetwork[]>();
			_networkChangedLock = new object();
		}

		public override void PushProgress(IWorker worker)
		{
			// first iteration of new epoch complete
			if (worker.LocalEpochNumber > EpochNumber && worker.LocalIterationNumber == 1)
			{
				PushEpochNetwork(worker);
			}

			// TODO invoke passive hooks for time steps (pass new epoch / new iteration as params? own methods?)
		}

		public override void PullProgress(IWorker worker)
		{
			// before first iteration of new epoch or network has not been initialised yet
			if (worker.LocalEpochNumber < EpochNumber && worker.LocalIterationNumber == 0 || worker.LocalNetwork == null)
			{
				worker.LocalNetwork = PullNetwork();
			}
		}

		protected virtual INetwork PullNetwork()
		{
			if (Network == null)
			{
				throw new InvalidOperationException($"Cannot pull network before assigning a network to operator {this}.");
			}

			lock (_networkChangedLock)
			{
				return (INetwork) Network.DeepCopy();
			}
		}

		protected virtual void PushEpochNetwork(IWorker worker)
		{
			bool allNetworksForEpochPushed;

			lock (_pushedNetworks)
			{
				INetwork[] networks = _pushedNetworks.TryGetValue(worker.LocalEpochNumber, () => new INetwork[WorkerCount]);
				if (!networks.AddToNextNull(worker.LocalNetwork.DeepCopy()))
				{
					throw new InvalidOperationException($"Too many workers trying to push their network, worker {worker} attempted to push his network but {WorkerCount} workers already pushed their network for epoch {worker.LocalEpochNumber}.");
				}

				allNetworksForEpochPushed = _pushedNetworks[worker.LocalEpochNumber][WorkerCount - 1] != null;
			}

			Logger.Info($"Worker {worker.GetType()} pushed its network for the epoch {worker.LocalEpochNumber}.");

			if (allNetworksForEpochPushed)
			{
				EpochNumber++;

				Logger.Info($"All workers (total of {WorkerCount}) are done with epoch {worker.LocalEpochNumber} in operator {this} and have pushed their network progress for this epoch.");
				Logger.Info($"Merging local pushed networks from all workers (total of {WorkerCount}) into global network of operator {this}...");

				lock (_networkChangedLock)
				{
					NetworkMerger.Merge(Network, _pushedNetworks[worker.LocalEpochNumber]);
				}

				Logger.Info($"Done merging local pushed networks from all workers (total of {WorkerCount}) into global network of operator {this}.");
			}
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
			Logger.Info($"Starting worker {worker} in operator {this}...");

			worker.Start();
		}

		protected override void RunWorkerOnce(IWorker worker)
		{
			Logger.Debug($"Running worker {worker} once in operator {this}...");

			worker.RunOnce();
		}

		protected override void PauseWorker(IWorker worker)
		{
			Logger.Debug($"Signalling pause to worker {worker} in operator {this}...");

			worker.SignalPause();
		}

		protected override void ResumeWorker(IWorker worker)
		{
			Logger.Debug($"Signalling resume to worker {worker} in operator {this}...");

			worker.SignalResume();
		}

		protected override void StopWorker(IWorker worker)
		{
			Logger.Info($"Stopping worker {worker} in operator {this}...");

			worker.SignalStop();
		}
	}
}