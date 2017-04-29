﻿/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Training.Hooks.Accumulators;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A hook that logs the accumulated / averaged value(s) of a certain identifier.
	/// </summary>
	[Serializable]
	public class AccumulatedValueReporterHook : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		///	Create a hook that fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		/// <param name="reportEpochIteration">Indicate whether or not to report the current epoch and iteration in addition to the values.</param>
		public AccumulatedValueReporterHook(string valueIdentifier, ITimeStep timestep, bool reportEpochIteration = false) : this(new[] { valueIdentifier }, timestep, reportEpochIteration: reportEpochIteration) { }

		/// <summary>
		/// Create a hook that conditionally (extrema criteria) fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		/// <param name="target">The extrema criteria target.</param>
		public AccumulatedValueReporterHook(string valueIdentifier, ITimeStep timestep, ExtremaTarget target) : this(new[] {valueIdentifier}, timestep)
		{
			On(new ExtremaCriteria(GetAccumulatedIdentifier(valueIdentifier), target));
		}

		/// <summary>
		/// Create a hook that conditionally (threshold criteria) fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		/// <param name="threshold">The threshold to compare against.</param>
		/// <param name="target">The threshold criteria comparison target.</param>
		/// <param name="fireContinously">If the value should be reported every time step the criteria is satisfied (or just once).</param>
		public AccumulatedValueReporterHook(string valueIdentifier, ITimeStep timestep, double threshold, ComparisonTarget target, bool fireContinously = true) : this(new[] { valueIdentifier }, timestep)
		{
			On(new ThresholdCriteria(GetAccumulatedIdentifier(valueIdentifier), target, threshold, fireContinously));
		}

		///  <summary>
		/// 	Create a hook that fetches a given amount of values (i.e. registry identifiers) at a given <see cref="ITimeStep"/>.
		///  </summary>
		///  <param name="valueIdentifiers">The values that will be fetched (i.e. registry identifiers). E.g. <c>"optimiser.cost_total"</c>, ...</param>
		///  <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		/// <param name="reportEpochIteration">Indicate whether or not to report the current epoch and iteration in addition to the values.</param>
		public AccumulatedValueReporterHook(string[] valueIdentifiers, ITimeStep timestep, bool averageValues = false, bool reportEpochIteration = false) : base(timestep, valueIdentifiers)
		{
			if (valueIdentifiers.Length == 0) throw new ArgumentException("Value identifiers cannot be empty (it's the whole point of this hook).");

			DefaultTargetMode = TargetMode.Local;

			string[] accumulatedIdentifiers = new string[valueIdentifiers.Length];
			Dictionary<string, object> valueBuffer = new Dictionary<string, object>(valueIdentifiers.Length);

			for (int i = 0; i < valueIdentifiers.Length; i++)
			{
				string value = valueIdentifiers[i];

				accumulatedIdentifiers[i] = GetAccumulatedIdentifier(value);

				// TODO let caller decide if it's a number (double) / something else

				int resetEvery, resetInterval;

				if (timestep.TimeScale == TimeScale.Iteration)
				{
					resetEvery = timestep.Interval;
					resetInterval = -1;
				}
				else
				{
					resetEvery = -1;
					resetInterval = 0;
				}

				RequireHook(new NumberAccumulatorHook(value, accumulatedIdentifiers[i], Utils.TimeStep.Every(1, TimeScale.Iteration), averageValues, resetEvery, resetInterval));

				valueBuffer.Add(value, null);
			}

			ParameterRegistry["value_identifiers"] = valueIdentifiers;
			ParameterRegistry["accumulated_identifiers"] = accumulatedIdentifiers;
			ParameterRegistry["value_buffer"] = valueBuffer;
			ParameterRegistry["report_epoch_iteration"] = reportEpochIteration;
		}

		private static string GetAccumulatedIdentifier(string value)
		{
			return "shared." + value.Replace('.', '_') + "_accumulated";
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			string[] accumulatedIdentifiers = ParameterRegistry.Get<string[]>("accumulated_identifiers");
			string[] valueIdentifiers = ParameterRegistry.Get<string[]>("value_identifiers");

			IDictionary<string, object> valuesByIdentifier = ParameterRegistry.Get<IDictionary<string, object>>("value_buffer");

			for (int i = 0; i < valueIdentifiers.Length; i++)
			{
				// TODO let callee decide if it's a number (double) / something else
				object value = resolver.ResolveGetSingle<double>(accumulatedIdentifiers[i]);

				valuesByIdentifier[valueIdentifiers[i]] = value;
			}

			ReportValues(valuesByIdentifier, ParameterRegistry.Get<bool>("report_epoch_iteration"), registry.Get<int>("epoch"), registry.Get<int>("iteration"));
		}

		/// <summary>
		/// Report the values for a certain epoch / iteration.
		/// Note: By default, this method writes to the logger. If you want to report to anywhere else, overwrite this method.
		/// </summary>
		/// <param name="valuesByIdentifier">The values by their identifier.</param>
		/// <param name="reportEpochIteration">A boolean indicating whether or not to report the current epoch / iteration.</param>
		/// <param name="epoch">The current epoch.</param>
		/// <param name="iteration">The current iteration.</param>
		protected virtual void ReportValues(IDictionary<string, object> valuesByIdentifier, bool reportEpochIteration, int epoch, int iteration)
		{
			string formattedValues = string.Join(", ", valuesByIdentifier.Select(pair => $"{pair.Key} = {pair.Value}"));

			_logger.Info((reportEpochIteration ? $"epoch {epoch} / iteration {iteration}: " : "") + formattedValues);
		}
	}
}
