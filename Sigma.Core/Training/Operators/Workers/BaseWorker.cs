/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
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
		private readonly IRegistry _bufferRegistry;

		/// <summary>
		///		The time scale countdowns per passive hook (passive hooks are managed by the operator).
		/// </summary>
		protected readonly IDictionary<IHook, ITimeStep> LocalActiveHookTimeSteps;


		protected BaseWorker(IOperator @operator) : this(@operator, @operator.Handler)
		{
		}

		protected BaseWorker(IOperator @operator, IComputationHandler handler)
		{
			Operator = @operator;
			Handler = handler;
			LocalActiveHookTimeSteps = new Dictionary<IHook, ITimeStep>();
			_bufferHooksToInvoke = new HashSet<IHook>();
			_bufferRegistry = new Registry();
		}

		public abstract void Start();
		public abstract void RunOnce();
		public abstract void SignalPause();
		public abstract void SignalResume();
		public abstract void SignalStop();

		public void InvokeTimeScaleEvent(TimeScale timeScale)
		{
			var activeHooks = Operator.Trainer.ActiveHooks;

			Operator.EjectTimeScaleEvent(timeScale, this, activeHooks, LocalActiveHookTimeSteps, _bufferHooksToInvoke);
			MarkDeadHooks(activeHooks, LocalActiveHookTimeSteps);

			Operator.PopulateWorkerRegistry(_bufferRegistry, this);

			foreach (IActiveHook hook in activeHooks)
			{
				hook.Invoke(_bufferRegistry);
			}
		}

		protected void MarkDeadHooks(IEnumerable<IActiveHook> hooks, IDictionary<IHook, ITimeStep> localTimeSteps)
		{
			foreach (IHook hook in _bufferHooksToInvoke)
			{
				IActiveHook asActiveHook = hook as IActiveHook;

				if (asActiveHook == null)
				{
					throw new InvalidOperationException($"Internal error: Buffered hooks to invoke in worker may only contain active hooks but hook {hook} could not be cast accordingly.");
				}

				if (LocalActiveHookTimeSteps[hook].LocalLiveTime == 0)
				{
					Operator.MarkHookDead(asActiveHook, this);
				}
			}
		}
	}
}