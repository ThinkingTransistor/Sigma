/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Training.Hooks.Processors;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Stoppers
{
	/// <summary>
	/// A stop training hook that will stop training when certain conditions are met (and reports so to the logger).
	/// </summary>
	[Serializable]
	public class StopTrainingHook : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook that will stop training when a certain custom criteria is met.
		/// Note: The time step defaults to every epoch for convenience, if that is not enough, use <see cref="StopTrainingHook(HookInvokeCriteria, TimeStep)"/>.
		/// </summary>
		/// <param name="criteria">The custom criteria at which training shall be stopped.</param>
		public StopTrainingHook(HookInvokeCriteria criteria) : this(criteria, new TimeStep(TimeScale.Epoch, 1))
		{
		}

		/// <summary>
		/// Create a hook that will stop training when a certain custom criteria is met with a custom time step.
		/// </summary>
		/// <param name="criteria">The custom criteria at which training shall be stopped.</param>
		/// <param name="timeStep">The time step of this hook (see <see cref="IHook"/>).</param>
		public StopTrainingHook(HookInvokeCriteria criteria, TimeStep timeStep) : this(timeStep)
		{
			On(criteria);
		}

		/// <summary>
		/// Create a hook that will stop training after <see cref="atEpoch"/> epochs.
		/// </summary>
		public StopTrainingHook(int atEpoch) : this(new ThresholdCriteria("epoch", ComparisonTarget.Equals, atEpoch, false))
		{
		}

		/// <summary>
		/// Internal constructor for custom training hook derivatives with additional requirements.
		/// </summary>
		/// <param name="timeStep"></param>
		protected StopTrainingHook(TimeStep timeStep) : base(timeStep)
		{
			DefaultTargetMode = TargetMode.Global;
			InvokePriority = 10000; // typically the training should be stopped after all other hooks have been invoked
									//  (hooks would be invoked anyway, it just looks cleaner)
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			_logger.Info($"Stopping training because condition {InvokeCriteria} was met.");

			Operator.SignalStop();
		}
	}

	/// <summary>
	/// A utility hook that stops training if a certain parameter does not improve (i.e. increase / decrease as specified) for a certain amount of time steps. 
	/// </summary>
	[Serializable]
	public class EarlyStopperHook : StopTrainingHook
	{
		/// <summary>
		/// Create an early stopper hook for a certain parameter that stops training if the parameter does not improve for <see cref="patience"/> time steps.
		/// Improve means reach a new <see cref="ExtremaTarget.Max"/> by default or a new <see cref="ExtremaTarget.Min"/> if specified in the <see cref="target"/>.
		/// </summary>
		/// <param name="parameter">The parameter identifier.</param>
		/// <param name="patience">The patience (number of sequential decreases of the parameter, i.e. how many times the training can let us down before we call it a day).</param>
		/// <param name="target">The target for the given value (i.e. should it be a big or a small value).</param>
		public EarlyStopperHook(string parameter, int patience, ExtremaTarget target = ExtremaTarget.Max) : base(new TimeStep(TimeScale.Epoch, 1))
		{
			string accumulatedParameter = "shared." + parameter.Replace('.', '_') + "_accumulated";
			NumberAccumulatorHook accumulator = new NumberAccumulatorHook(parameter, accumulatedParameter, Utils.TimeStep.Every(1, TimeScale.Iteration));

			RequireHook(accumulator);
			On(new ExtremaCriteria(accumulatedParameter, target).Negate().Repeated(patience, withoutInterruption: true));
		}
	}
}
