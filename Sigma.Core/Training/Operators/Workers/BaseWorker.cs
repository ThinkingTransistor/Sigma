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

		/// <summary>
		///		The time scale countdowns per passive hook (passive hooks are managed by the operator).
		/// </summary>
		protected readonly IDictionary<IHook, int> ActiveHooksTimeScaleCountdowns;

		protected BaseWorker(IOperator @operator) : this(@operator, @operator.Handler)
		{
		}

		protected BaseWorker(IOperator @operator, IComputationHandler handler)
		{
			Operator = @operator;
			Handler = handler;
			ActiveHooksTimeScaleCountdowns = new Dictionary<IHook, int>();
		}

		public abstract void Start();
		public abstract void RunOnce();
		public abstract void SignalPause();
		public abstract void SignalResume();
		public abstract void SignalStop();

		/// <summary>
		/// Invoke active hooks for a certain time scale with a certain worker.
		/// </summary>
		/// <param name="timeScale">The time scale.</param>
		public void InvokeActiveHooks(TimeScale timeScale)
		{
			Operator.InvokeActiveHooks(timeScale, this, ActiveHooksTimeScaleCountdowns);
		}
	}
}