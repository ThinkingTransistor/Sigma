/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Utils;
using System;
using System.ComponentModel;

namespace Sigma.Core.Training.Hooks
{
	/// <summary>
	/// A base hook invoke criteria for additionally limiting hook invocations to certain criteria.
	/// </summary>
	public abstract class HookInvokeCriteria
	{
		internal string[] RequiredRegistryEntries { get; }
		internal bool[] SimpleDirectEntries { get; }
		internal IRegistry ParameterRegistry { get; }

		protected HookInvokeCriteria(params string[] requiredRegistryEntries)
		{
			if (requiredRegistryEntries == null) throw new ArgumentNullException(nameof(requiredRegistryEntries));

			SimpleDirectEntries = new bool[requiredRegistryEntries.Length];
			for (var i = 0; i < requiredRegistryEntries.Length; i++)
			{
				SimpleDirectEntries[i] = !requiredRegistryEntries[i].Contains(".");
			}

			RequiredRegistryEntries = requiredRegistryEntries;
			ParameterRegistry = new Registry();
			ParameterRegistry["required_registry_entries"] = requiredRegistryEntries;
		}

		public abstract bool CheckCriteria(IRegistry registry, IRegistryResolver resolver);

		internal virtual bool FunctionallyEquals(HookInvokeCriteria other)
		{
			return GetType() == other.GetType() && ParameterRegistry.RegistryContentEquals(other.ParameterRegistry);
		}
	}

	/// <summary>
	/// A threshold criteria that fires when a certain threshold is reached (once or continuously as specified). 
	/// </summary>
	public class ThresholdCriteria : HookInvokeCriteria
	{
		public ThresholdCriteria(string parameter, ComparisonTarget target, double thresholdValue, bool fireContinously = true) : base(parameter)
		{
			if (parameter == null) throw new ArgumentNullException(nameof(parameter));

			ParameterRegistry["parameter_identifier"] = parameter;
			ParameterRegistry["target"] = target;
			ParameterRegistry["threshold_value"] = thresholdValue;
			ParameterRegistry["fire_continously"] = fireContinously;
			ParameterRegistry["last_check_met"] = false;
			ParameterRegistry["threshold_reached"] = false;
		}

		public override bool CheckCriteria(IRegistry registry, IRegistryResolver resolver)
		{
			string parameter = ParameterRegistry.Get<string>("parameter_identifier");
			double value = SimpleDirectEntries[0] ? registry.Get<double>(parameter) : resolver.ResolveGetSingle<double>(parameter);
			bool thresholdReached = _InternalThresholdReached(value, ParameterRegistry.Get<double>("threshold_value"), ParameterRegistry.Get<ComparisonTarget>("target"));
			bool fire = thresholdReached && (!ParameterRegistry.Get<bool>("last_check_met") || ParameterRegistry.Get<bool>("fire_continously"));

			ParameterRegistry["last_check_met"] = thresholdReached;

			return fire;
		}

		private bool _InternalThresholdReached(double value, double threshold, ComparisonTarget target)
		{
			switch (target)
			{
				case ComparisonTarget.Equals:
					return value == threshold;
				case ComparisonTarget.GreaterThanEquals:
					return value >= threshold;
				case ComparisonTarget.GreaterThan:
					return value > threshold;
				case ComparisonTarget.SmallerThanEquals:
					return value <= threshold;
				case ComparisonTarget.SmallerThan:
					return value < threshold;
				default:
					throw new ArgumentOutOfRangeException($"Comparison target is out of range ({target}), use the provided {nameof(ComparisonTarget)} enum.");
			}
		}
	}

	/// <summary>
	/// An extrema criteria that fires when a value has reached a new extrema (min / max).
	/// </summary>
	public class ExtremaCriteria : HookInvokeCriteria
	{
		public ExtremaCriteria(string parameter, ExtremaTarget target) : base(parameter)
		{
			if (parameter == null) throw new ArgumentNullException(nameof(parameter));
			if (!Enum.IsDefined(typeof(ExtremaTarget), target)) throw new InvalidEnumArgumentException(nameof(target), (int) target, typeof(ExtremaTarget));

			ParameterRegistry["parameter_identifier"] = parameter;
			ParameterRegistry["target"] = target;
			ParameterRegistry["current_extremum"] = double.NaN;
		}

		public override bool CheckCriteria(IRegistry registry, IRegistryResolver resolver)
		{
			ExtremaTarget target = ParameterRegistry.Get<ExtremaTarget>("target");
			string parameter = ParameterRegistry.Get<string>("parameter_identifier");
			double value = SimpleDirectEntries[0] ? registry.Get<double>(parameter) : resolver.ResolveGetSingle<double>(parameter);
			double currentExtremum = ParameterRegistry.Get<double>("current_extremum");
			bool reachedExtremum = target == ExtremaTarget.Min && value < currentExtremum || target == ExtremaTarget.Max && value > currentExtremum;

			if (double.IsNaN(currentExtremum) || reachedExtremum)
			{
				ParameterRegistry["current_extremum"] = value;

				return true;
			}

			return false;
		}
	}

	/// <summary>
	/// A comparison target for conditional invokes. 
	/// </summary>
	public enum ComparisonTarget { GreaterThan, GreaterThanEquals, Equals, SmallerThanEquals, SmallerThan } 

	/// <summary>
	/// An extrema target for conditional invokes.
	/// </summary>
	public enum ExtremaTarget { Max, Min}
}
