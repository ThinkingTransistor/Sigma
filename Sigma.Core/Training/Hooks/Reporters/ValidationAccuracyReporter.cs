using System;
using log4net;
using Sigma.Core.Training.Hooks.Scorers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	public class ValidationAccuracyReporter : BaseHook
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="validationIteratorName">The name of the validation data iterator to use (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		public ValidationAccuracyReporter(string validationIteratorName, ITimeStep timestep) : base(timestep)
		{
			DefaultTargetMode = TargetMode.Global;

			RequireHook(new ValidationAccuracyScorer(validationIteratorName, "shared.validation_accuracy_top1", timestep));
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void Invoke(IRegistry registry, IRegistryResolver resolver)
		{
			double score = resolver.ResolveGetSingle<double>("shared.validation_accuracy_top1");

			_logger.Info($"Validation accuracy: {score}");
		}
	}
}
