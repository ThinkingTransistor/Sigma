/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Training.Hooks.Processors;
using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A running time reporter that reports the running time of a certain time scale events.
	/// </summary>
	[Serializable]
	public class RunningTimeReporter : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timeStep">The time step.</param>
		/// <param name="averageSpan">The interval span to average over.</param>
		public RunningTimeReporter(ITimeStep timeStep, int averageSpan = 4) : base(timeStep)
		{
			DefaultTargetMode = TargetMode.Global;

			string baseResultKey = $"shared.time_{timeStep.TimeScale}";
			RequireHook(new RunningTimeProcessorHook(timeStep.TimeScale, averageSpan, baseResultKey));

			ParameterRegistry.Set("base_result_key", baseResultKey, typeof(string));
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			string baseResultKey = ParameterRegistry.Get<string>("base_result_key");

			long lastTime = resolver.ResolveGetSingleWithDefault(baseResultKey + "_last", -1L);
			long averageTime = resolver.ResolveGetSingleWithDefault(baseResultKey + "_average", -1L);

			Report(TimeStep.TimeScale, lastTime, averageTime);
		}

		/// <summary>
		/// Report the frequency of a time scale event.
		/// </summary>
		/// <param name="timeScale">The time scale event that just occurred.</param>
		/// <param name="lastTime">The elapsed time to the last occurrence.</param>
		/// <param name="averageTime">The average time between occurrences.</param>
		protected virtual void Report(TimeScale timeScale, long lastTime, long averageTime)
		{
			if (lastTime != -1 && averageTime != -1)
			{
				TimeUnit unit;
				var inverseTime = PrintUtils.GetInverseTime(averageTime, out unit);

				string lowerCaseTimescale = timeScale.ToString().ToLower();

				_logger.Info($"{inverseTime:###.0} {lowerCaseTimescale}s/{unit.GetTimeUnitInFormat(TimeUnitFormat.Minimum)} (time/{lowerCaseTimescale}: average {PrintUtils.FormatTimeSimple(averageTime)}, last {PrintUtils.FormatTimeSimple(lastTime)})");
			}
		}
	}
}
