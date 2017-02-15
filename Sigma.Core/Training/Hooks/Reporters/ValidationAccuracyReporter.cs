using System;
using System.Collections.Generic;
using log4net;
using Sigma.Core.Training.Hooks.Scorers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// This hook reports the validation accuracy of given tops. 
	/// </summary>
	public class ValidationAccuracyReporter : BaseHook
	{
		/// <summary>
		/// The tops that will be reported.
		/// 
		/// E.g. 1,3,5 -> report the first, the third and the fifth best (accuracy).
		/// </summary>
		public int[] Tops { get; }

		/// <summary>
		///	Top values are stored in this dictionary (in order to prevent repeated allocation).
		/// </summary>
		private readonly IDictionary<int, double> _topDictionary;

		/// <summary>
		/// The base validation accuracy top identifier (without the top number) used to get the top value.
		/// </summary>
		private const string ValidationAccuracyTopIdentifier = "shared.validation_accuracy_top";

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
			Tops = tops;

			DefaultTargetMode = TargetMode.Global;

			_topDictionary = new Dictionary<int, double>(tops.Length);

			foreach (int top in tops)
			{
				string currentTop = ValidationAccuracyTopIdentifier + top;
				_topDictionary.Add(top, double.NaN);

				RequireHook(new ValidationAccuracyScorer(validationIteratorName, currentTop, timestep));
			}
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			foreach (int top in Tops)
			{
				_topDictionary[top] = resolver.ResolveGetSingle<double>(ValidationAccuracyTopIdentifier + top);
			}
			
			Report(_topDictionary);
		}

		/// <summary>
		/// Execute the report for every given top. 
		/// </summary>
		/// <param name="data">The mapping between the tops specified in the constructor and the score of the top.</param>
		public virtual void Report(IDictionary<int, double> data)
		{
			_logger.Info($"Validation accuracy: top({string.Join(";", data.Keys)}) = {string.Join(";", data.Values)}");
		}
	}
}
