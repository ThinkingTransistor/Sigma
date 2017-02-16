/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Stoppers
{
	/// <summary>
	/// A stop training hook that will stop training when certain conditions are met.
	/// </summary>
	public class StopTrainingHook : BaseHook
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook that will stop training when a certain custom criteria is met.
		/// </summary>
		/// <param name="criteria">The custom criteria at which training shall be stopped.</param>
		public StopTrainingHook(HookInvokeCriteria criteria) : base(new TimeStep(TimeScale.Epoch, 1))
		{
			DefaultTargetMode = TargetMode.Global;
			InvokePriority = 10000; // typically the training should be stopped after all other hooks have been invoked
									//  (hooks would be invoked anyway, it just looks cleaner)

			On(criteria);
		}

		/// <summary>
		/// Create a hook that will stop training after <see cref="atEpoch"/> epochs.
		/// </summary>
		public StopTrainingHook(int atEpoch) : this(new ThresholdCriteria("epoch", ComparisonTarget.Equals, atEpoch, false))
		{
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
}
