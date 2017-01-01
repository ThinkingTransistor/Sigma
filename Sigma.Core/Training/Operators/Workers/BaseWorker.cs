/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Handlers;

namespace Sigma.Core.Training.Operators.Workers
{
	public abstract class BaseWorker : IWorker
	{
		public IOperator Operator { get; }
		public TrainingState State { get; } = TrainingState.None;
		public IComputationHandler Handler { get; }
		public INetwork LocalNetwork { get; set; }

		public int LocalIterationNumber { get; protected set; }

		protected BaseWorker(IOperator @operator) : this(@operator, @operator.Handler) { }

		protected BaseWorker(IOperator @operator, IComputationHandler handler)
		{
			Operator = @operator;
			Handler = handler;
		}

		public abstract void Start();
		public abstract void SignalPause();
		public abstract void SignalResume();
		public abstract void SignalStop();
	}
}