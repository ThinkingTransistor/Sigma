/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu
{
	public class CpuMultithreadedOperator : IOperator
	{
		public SigmaEnvironment Sigma { get; set; }
		public IComputationHandler Handler { get; }
		public ITrainer Trainer { get; }
		public INetwork Network { get; }
		public int WorkerCount { get; }

		public void AttachHook(IHook hook)
		{
			throw new System.NotImplementedException();
		}

		public void DetachHook(IHook hook)
		{
			throw new System.NotImplementedException();
		}

		public void Start()
		{
			throw new System.NotImplementedException();
		}

		public void SignalPause()
		{
			throw new System.NotImplementedException();
		}

		public void SignalResume()
		{
			throw new System.NotImplementedException();
		}

		public void SignalStop()
		{
			throw new System.NotImplementedException();
		}
	}
}
