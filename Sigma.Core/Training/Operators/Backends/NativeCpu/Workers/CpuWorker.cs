using System;
using System.Collections.Generic;
using System.Threading;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Operators.Workers;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseWorker
	{
		private ILog Logger => _logger ?? (_logger = LogManager.GetLogger(GetType()));
		private ILog _logger;

		private IEnumerator<IDictionary<string, INDArray>> _epochBlockYield;
		private double _totalEpochCost;

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

			var blockYieldEnumerable = LocalTrainingDataIterator?.Yield(Operator.Handler, Operator.Sigma);

			if (blockYieldEnumerable == null)
			{
				_logger.Warn($"Unable to yield block enumerable from local training iterator {LocalTrainingDataIterator} in worker {this}");

				return;
			}

			_epochBlockYield = blockYieldEnumerable.GetEnumerator();

			Logger.Debug($"Done initialising worker {this}.");
		}

		protected override void DoWork()
		{
			if (_epochBlockYield == null)
			{
				throw new InvalidOperationException($"Unable to do work in worker {this}, worker was not initialised successfully (epoch yield is null).");
			}

			// no more blocks in this yield, therefore epoch is done
			if (!_epochBlockYield.MoveNext())
			{
				Logger.Info($"Completed epoch {LocalEpochNumber + 1} at iteration {LocalIterationNumber} in worker {this}.");
				Logger.Error($"Total epoch cost of {_totalEpochCost}."); // TODO add proper progress reporting

				LocalEpochNumber++;
				LocalIterationNumber = 0;
				_epochBlockYield = LocalTrainingDataIterator.Yield(Operator.Handler, Operator.Sigma).GetEnumerator();
				_epochBlockYield.MoveNext();
				_totalEpochCost = 0.0;
			}

			if (_epochBlockYield.Current == null)
			{
				throw new InvalidOperationException($"Unable to do work in worker {this} when current epoch block yield is null.");
			}

			Operator.PullProgress(this);

			Operator.Trainer.ProvideExternalData(LocalNetwork, _epochBlockYield.Current);
			Operator.Trainer.RunTrainingIteration(LocalNetwork, LocalOptimiser, Operator.Handler);

			_totalEpochCost += (float) LocalOptimiser.Registry.Get<INumber>("total_cost").Value;

			//_logger.Debug($"Worker {this} done with iteration {LocalIterationNumber} in epoch {LocalEpochNumber} at cost:\t{LocalOptimiser.Registry.Get<INumber>("total_cost").Value}");

			LocalIterationNumber++;

			// push progress for this iteration
			Operator.PushProgress(this);
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