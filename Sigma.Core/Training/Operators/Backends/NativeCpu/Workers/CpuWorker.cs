using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseCpuWorker
	{
		private ILog Logger => _logger ?? (_logger = LogManager.GetLogger(GetType()));
		private ILog _logger;

		private IEnumerator<IDictionary<string, INDArray>> _epochBlockYield;

		public CpuWorker(IOperator @operator) : base(@operator)
		{
		}

		public CpuWorker(IOperator @operator, IComputationHandler handler) : base(@operator, handler)
		{
		}

		public CpuWorker(IOperator @operator, IComputationHandler handler, ThreadPriority priority) : base(@operator, handler, priority)
		{
		}

		protected override void Initialise()
		{
			Logger.Debug($"Initialising worker {this}...");

			_epochBlockYield = LocalTrainingDataIterator?.Yield(Operator.Handler, Operator.Sigma).GetEnumerator();

			Logger.Debug($"Done initialising worker {this}.");
		}

		protected override void DoWork()
		{
			if (LocalNetwork == null)
			{
				Operator.PullProgress(this);
			}

			if (_epochBlockYield == null)
			{
				throw new InvalidOperationException($"Cannot work in worker {this} because the epoch block yield enumerator was not successfully initialised.");
			}

			if (!_epochBlockYield.MoveNext())
			{
				Logger.Info($"Completed epoch {LocalEpochNumber + 1} at iteration {LocalIterationNumber} in worker {this}.");

				LocalEpochNumber++;
				LocalIterationNumber = 0;
				_epochBlockYield = LocalTrainingDataIterator.Yield(Operator.Handler, Operator.Sigma).GetEnumerator();

				// epoch done, push progress for this epoch
				Operator.PushProgress(this);

				if (Operator.EpochNumber >= LocalEpochNumber)
				{
					Operator.PullProgress(this);
				}
			}

			Operator.Trainer.ProvideExternalData(LocalNetwork, _epochBlockYield.Current);
			Operator.Trainer.RunTrainingIteration(LocalNetwork, LocalOptimiser, Operator.Handler);

			LocalIterationNumber++;
		}

		protected override void OnPause()
		{
		}

		protected override void OnResume()
		{
		}

		protected override void OnStop()
		{
		}
	}
}