/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Workers
{
	public abstract class BaseWorker : IWorker
	{
		public IOperator Operator { get; }
		public ExecutionState State { get; protected set; } = ExecutionState.None;
		public IComputationHandler Handler { get; }
		public INetwork LocalNetwork { get; set; }
		public IDataIterator LocalTrainingDataIterator { get; set; }
		public IOptimiser LocalOptimiser { get; set; }

		public int LocalEpochNumber { get; protected set; }
		public int LocalIterationNumber { get; protected set; }

		private readonly ISet<IHook> _bufferHooksToInvoke;

		/// <summary>
		///		The time scale countdowns per passive hook (passive hooks are managed by the operator).
		/// </summary>
		protected readonly IDictionary<IHook, TimeStep> LocalActiveHookTimeSteps;


		protected BaseWorker(IOperator @operator) : this(@operator, @operator.Handler)
		{
		}

		protected BaseWorker(IOperator @operator, IComputationHandler handler)
		{
			Operator = @operator;
			Handler = handler;
			LocalActiveHookTimeSteps = new Dictionary<IHook, TimeStep>();
			_bufferHooksToInvoke = new HashSet<IHook>();
		}

		public abstract void Start();
		public abstract void RunOnce();
		public abstract void SignalPause();
		public abstract void SignalResume();
		public abstract void SignalStop();

		public void EjectAndInvokeTimeScaleEvent(TimeScale timeScale)
		{
			Operator.EjectTimeScaleEvent(timeScale, this, Operator.Trainer.ActiveHooks, LocalActiveHookTimeSteps, _bufferHooksToInvoke);

			// TODO actually invoke hooks, maybe find better method name or split in two methods
		}
	}
}