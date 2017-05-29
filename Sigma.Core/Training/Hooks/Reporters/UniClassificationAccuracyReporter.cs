/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using log4net;
using Sigma.Core.Training.Hooks.Scorers;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Reporters
{
	/// <summary>
	/// This hook reports the uni-class classification accuracy of targets with a certain threshold.
	/// </summary>
	[Serializable]
	public class UniClassificationAccuracyReporter : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		/// <param name="threshold">The threshold above which predictions are treated as 1 and below as 0.</param>
		public UniClassificationAccuracyReporter(string validationIteratorName, double threshold, ITimeStep timestep) : this(validationIteratorName, threshold, threshold, timestep)
		{
		}

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		/// <param name="lowerThreshold">The lower threshold, below which predictions are treated as 0.</param>
		/// <param name="upperThreshold">The upper threshold, above which predictions are treated as 1.</param>
		public UniClassificationAccuracyReporter(string validationIteratorName, double lowerThreshold, double upperThreshold, ITimeStep timestep) : base(timestep)
		{
			DefaultTargetMode = TargetMode.Global;
			InvokePriority = -100;

			RequireHook(new UniClassificationAccuracyScorer(validationIteratorName, "shared.classification_accuracy", lowerThreshold, upperThreshold, timestep));
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			double accuracy = resolver.ResolveGetSingle<double>("shared.classification_accuracy");

			Report(accuracy);
		}

		/// <summary>
		/// Report the given classification accuracy.
		/// </summary>
		/// <param name="accuracy">The accuracy.</param>
		protected void Report(double accuracy)
		{
			_logger.Info($"accuracy = {accuracy:0.000}");
		}
	}
}
