/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Training.Hooks.Scorers;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// This hook reports the validation accuracy of given tops (typically for classification tasks). 
	/// </summary>
	[Serializable]
	public class ValidationAccuracyReporter : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="validationIteratorName">The name of the validation data iterator to use (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		/// <param name="tops">The tops that will get reported.</param>
		public ValidationAccuracyReporter(string validationIteratorName, ITimeStep timestep, params int[] tops) : base(timestep)
		{
			if (tops.Length == 0)
			{
				throw new ArgumentException("Value cannot be an empty collection.", nameof(tops));
			}

			DefaultTargetMode = TargetMode.Global;
		    InvokePriority = -100;
			ParameterRegistry["tops"] = tops;
			
			RequireHook(new ValidationAccuracyScorer(validationIteratorName, "shared.validation_accuracy_top", timestep, tops));
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			int[] tops = ParameterRegistry.Get<int[]>("tops");
			IDictionary<int, double> topDictionary = new Dictionary<int, double>();

			foreach (int top in tops)
			{
				topDictionary[top] = resolver.ResolveGetSingle<double>("shared.validation_accuracy_top" + top);
			}
			
			Report(topDictionary);
		}

		/// <summary>
		/// Execute the report for every given top. 
		/// </summary>
		/// <param name="data">The mapping between the tops specified in the constructor and the score of the top.</param>
		protected virtual void Report(IDictionary<int, double> data)
		{
			_logger.Info(string.Join(", ", data.Select(p => $"top{p.Key} = {p.Value}")));
		}
	}
}
