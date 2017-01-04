using System.Threading;
using Sigma.Core.Handlers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseCpuWorker
	{
		public CpuWorker(IOperator @operator) : base(@operator) { }

		public CpuWorker(IOperator @operator, IComputationHandler handler) : base(@operator, handler) { }

		public CpuWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority) : base(@operator, handler, priority) { }

		protected override void Initialise()
		{

		}

		protected override void DoWork()
		{

		}

		public override void RunTrainingIteration()
		{

		}

		protected override void Pause()
		{

		}

		protected override void Resume()
		{

		}

		protected override void Stop()
		{

		}
	}
}