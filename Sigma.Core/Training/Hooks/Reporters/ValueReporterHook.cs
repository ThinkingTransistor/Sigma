/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using log4net;
using Sigma.Core.Training.Hooks.Accumulators;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// A hook that logs the current local value of each worker (e.g. cost) over a certain time period.
	/// </summary>
	public class ValueReporterHook : BaseHook
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public ValueReporterHook(string[] valueIdentifiers, ITimeStep timestep) : base(timestep, valueIdentifiers)
		{
			if (valueIdentifiers.Length == 0) throw new ArgumentException($"Value identifiers cannot be empty (it's the whole point of this hook).");

			ParameterRegistry["value_identifiers"] = valueIdentifiers;

			foreach (string value in valueIdentifiers)
			{
				// TODO let callee decide if it's a number / something else
				RequireHook(new NumberAccumulatorHook(value, value + "_accumulated", Utils.TimeStep.Every(1, TimeScale.Iteration)));
			}
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver"></param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			StringBuilder builder = new StringBuilder($"Epoch {registry["epoch"]} / iteration {registry["iteration"]}: ");

			string[] valueIdentifiers = ParameterRegistry.Get<string[]>("value_identifiers");

			builder.Append(string.Join(",", valueIdentifiers.Select(resolver.ResolveGetSingle<double>)));
		}
	}
}
