/* 
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
	/// A hook that logs the current local value of each worker (e.g. cost) over a certain time period.
	/// </summary>
	public class ValueReporterHook : BaseHook
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		///	Create a hook that fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueReporterHook(string valueIdentifier, ITimeStep timestep) : this(new[] { valueIdentifier }, timestep) { }

		/// <summary>
		/// Create a hook that conditionally (extrema criteria) fetches a given value (i.e. registry identifier) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifier">The value that will be fetched (i.e. registry identifier). E.g. <c>"optimiser.cost_total"</c></param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		/// <param name="target">The extrema criteria target.</param>
		public ValueReporterHook(string valueIdentifier, ITimeStep timestep, ExtremaTarget target) : this(new[] {valueIdentifier}, timestep)
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
		public ValueReporterHook(string valueIdentifier, ITimeStep timestep, double threshold, ComparisonTarget target, bool fireContinously = true) : this(new[] { valueIdentifier }, timestep)
		{
			On(new ThresholdCriteria(GetAccumulatedIdentifier(valueIdentifier), target, threshold, fireContinously));
		}

		/// <summary>
		///	Create a hook that fetches a given amount of values (i.e. registry identifiers) at a given <see cref="ITimeStep"/>.
		/// </summary>
		/// <param name="valueIdentifiers">The values that will be fetched (i.e. registry identifiers). E.g. <c>"optimiser.cost_total"</c>, ...</param>
		/// <param name="timestep">The <see cref="ITimeStep"/> the hook will executed on.</param>
		public ValueReporterHook(string[] valueIdentifiers, ITimeStep timestep) : base(timestep, valueIdentifiers)
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
				RequireHook(new NumberAccumulatorHook(value, accumulatedIdentifiers[i], Utils.TimeStep.Every(1, TimeScale.Iteration)));

				valueBuffer.Add(value, null);
			}

			ParameterRegistry["value_identifiers"] = valueIdentifiers;
			ParameterRegistry["accumulated_identifiers"] = accumulatedIdentifiers;
			ParameterRegistry["value_buffer"] = valueBuffer;
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

			ReportValues(valuesByIdentifier);
		}

		/// <summary>
		/// Report the values for a certain epoch / iteration.
		/// Note: By default, this method writes to the logger. If you want to report to anywhere else, overwrite this method.
		/// </summary>
		/// <param name="valuesByIdentifier">The values by their identifier.</param>
		protected virtual void ReportValues(IDictionary<string, object> valuesByIdentifier)
		{
			_logger.Info(string.Join(", ", valuesByIdentifier.Select(pair => $"{pair.Key} = {pair.Value}")));
		}
	}
}
