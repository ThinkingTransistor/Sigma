/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Threading;
using log4net;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Operators.Workers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Operators.Backends.NativeCpu.Workers
{
	public class CpuWorker : BaseWorker
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
				InvokeTimeScaleEvent(TimeScale.Epoch);

				Logger.Debug($"Completed epoch {LocalEpochNumber} at iteration {LocalIterationNumber} in worker {this}.");

				LocalEpochNumber++;
				LocalIterationNumber = 0;

				_epochBlockYield = LocalTrainingDataIterator.Yield(Operator.Handler, Operator.Sigma).GetEnumerator();
				_epochBlockYield.MoveNext();
			}

			if (_epochBlockYield.Current == null)
			{
				throw new InvalidOperationException($"Unable to do work in worker {this} because current epoch block yield is null.");
			}

			Operator.PullProgress(this);

			Operator.Trainer.ProvideExternalInputData(LocalNetwork, _epochBlockYield.Current);
			Operator.Trainer.RunTrainingIteration(LocalNetwork, LocalOptimiser, GetPopulatedBufferRegistry(), Operator.Handler);
			Operator.Trainer.ProvideExternalOutputData(LocalNetwork, _epochBlockYield.Current);

			InvokeTimeScaleEvent(TimeScale.Iteration);

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