/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Backends.NativeCpu.Workers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu
{
	public class CpuSinglethreadedOperator : CpuMultithreadedOperator
	{
		public CpuSinglethreadedOperator(SigmaEnvironment sigma, IComputationHandler handler, ITrainer trainer,
			INetwork network) : base(sigma, handler, trainer, network, 1)
		{
		}
	}

	public class CpuMultithreadedOperator : BaseOperator
	{
		public CpuMultithreadedOperator(SigmaEnvironment sigma, IComputationHandler handler, ITrainer trainer, INetwork network, int workerCount) : base(sigma, handler, trainer, network, workerCount)
		{
		}

		public override IWorker CreateWorker(IOperator @operator)
		{
			return new CpuWorker(@operator);
		}

		public override void StartWorker(IWorker worker)
		{
			worker.Start();
		}

		public override void PauseWorker(IWorker worker)
		{
			worker.SignalPause();
		}

		public override void ResumeWorker(IWorker worker)
		{
			worker.SignalResume();
		}

		public override void StopWorker(IWorker worker)
		{
			worker.SignalStop();
		}
	}
}
