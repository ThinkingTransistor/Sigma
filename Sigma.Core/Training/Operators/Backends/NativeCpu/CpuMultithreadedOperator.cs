/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Threading;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Training.Operators.Backends.NativeCpu.Workers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu
{
	/// <summary>
	/// This class is just an alias for the <see cref="CpuMultithreadedOperator"/>.
	/// On some cases it can be useful to have separate classes. 
	/// It runs only on a single thread.
	/// </summary>
	public class CpuSinglethreadedOperator : CpuMultithreadedOperator
	{
		public CpuSinglethreadedOperator(SigmaEnvironment sigma) : this(sigma, new CpuFloat32Handler())
		{
		}

		public CpuSinglethreadedOperator(SigmaEnvironment sigma, IComputationHandler handler) : base(sigma, handler, 1)
		{
		}

		public CpuSinglethreadedOperator(SigmaEnvironment sigma, IComputationHandler handler,
			ThreadPriority workerPriority) : base(sigma, handler, 1, workerPriority)
		{
		}
	}

	/// <summary>
	/// This is a multithreaded CPU-based operator. It will execute all 
	/// operations on the CPU. The tasks will be executed concurrently by the
	/// number of threads specified. 
	/// </summary>
	public class CpuMultithreadedOperator : BaseOperator
	{
		public ThreadPriority WorkerPriority { get; set; }

		public CpuMultithreadedOperator(SigmaEnvironment sigma, int workerCount)
			: this(sigma, new CpuFloat32Handler(), workerCount)
		{
		}

		public CpuMultithreadedOperator(SigmaEnvironment sigma, IComputationHandler handler, int workerCount)
			: this(sigma, handler, workerCount, ThreadPriority.Highest)
		{
		}

		public CpuMultithreadedOperator(SigmaEnvironment sigma, IComputationHandler handler, int workerCount,
			ThreadPriority workerPriority) : base(sigma, handler, workerCount)
		{
			WorkerPriority = workerPriority;
		}


		protected override IWorker CreateWorker()
		{
			return new CpuWorker(this, Handler, WorkerPriority);
		}

		protected override void StartWorker(IWorker worker)
		{
			worker.Start();
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
