/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Threading;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseWorker
	{
		protected Thread WorkerThread { get; set; }

		public CpuWorker(IOperator @operator) : base(@operator) { }

		public CpuWorker(IOperator @operator, IComputationHandler handler) : base(@operator, handler) { }

		public override void Start()
		{
			throw new System.NotImplementedException();
		}

		public override void SignalPause()
		{
			throw new System.NotImplementedException();
		}

		public override void SignalResume()
		{
			throw new System.NotImplementedException();
		}

		public override void SignalStop()
		{
			throw new System.NotImplementedException();
		}

		protected virtual void Update()
		{

		}
	}
}