/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Scorers
{
	/// <summary>
	/// A uni-class classification accuracy scorer that scores a single predicted target against the actual target with a certain threshold. 
	/// </summary>
	public class UniClassificationAccuracyScorer : BaseAccuracyScorer
	{
		/// <summary>
		/// Create a validation scorer hook for a certain validation iterator.
		/// Note:	The "external_default" output may not always be the actual final output in your model.
		///			If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="resultKey">The result key under which the result accuracy will be available.</param>
		/// <param name="timestep">The time step.</param>
		public UniClassificationAccuracyScorer(string validationIteratorName, string resultKey, ITimeStep timestep)
			: this(validationIteratorName, resultKey, 0.5, timestep)
		{
		}

		///  <summary>
		///  Create a validation scorer hook for a certain validation iterator.
		///  Note:	The "external_default" output may not always be the actual final output in your model.
		/// 			If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		///  </summary>
		///  <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		///  <param name="resultKey">The result key under which the result accuracy will be available.</param>
		/// <param name="threshold">The threshold above which predictions are treated as 1 and below as 0.</param>
		/// <param name="timestep">The time step.</param>
		public UniClassificationAccuracyScorer(string validationIteratorName, string resultKey, double threshold, ITimeStep timestep)
			: this(validationIteratorName, resultKey, threshold, threshold, timestep)
		{
		}

		/// <summary>
		/// Create a validation scorer hook for a certain validation iterator.
		/// Note:	The "external_default" output may not always be the actual final output in your model.
		/// 		If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		/// <param name="resultKey">The result key under which the result accuracy will be available.</param>
		/// <param name="lowerThreshold">The lower threshold, below which predictions are treated as 0.</param>
		/// <param name="upperThreshold">The upper threshold, above which predictions are treated as 1.</param>
		public UniClassificationAccuracyScorer(string validationIteratorName, string resultKey, double lowerThreshold, double upperThreshold, ITimeStep timestep)
			: base(validationIteratorName, timestep)
		{
			if (resultKey == null) throw new ArgumentNullException(nameof(resultKey));
			if (upperThreshold < lowerThreshold) throw new ArgumentException($"Lower threshold must be lower than upper threshold but lower was {lowerThreshold} and upper was {upperThreshold}.");

			ParameterRegistry["lower_threshold"] = lowerThreshold;
			ParameterRegistry["upper_threshold"] = upperThreshold;
			ParameterRegistry["result_key"] = resultKey;
		}

		/// <summary>
		/// Begin a validation scoring session.
		/// Reset the scoring here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreBegin(IRegistry registry, IRegistryResolver resolver)
		{
			ParameterRegistry["total_classifications"] = 0;
			ParameterRegistry["correct_classifications"] = 0;
		}

		/// <summary>
		/// Score an intermediate result (part of the entire validation dataset was used as specified by the validation iterator).
		/// </summary>
		/// <param name="predictions">The predictions.</param>
		/// <param name="targets">The targets.</param>
		/// <param name="handler">The computation handler.</param>
		protected override void ScoreIntermediate(INDArray predictions, INDArray targets, IComputationHandler handler)
		{
			predictions = handler.FlattenTimeAndFeatures(predictions);
			targets = handler.FlattenTimeAndFeatures(targets);

			if (predictions.Shape[1] != 1)
			{
				throw new InvalidOperationException($"Cannot score uni-class classification accuracy on targets with != 1 feature shape (feature shape length was {predictions.Shape[1]}).");
			}

			int totalClassifications = ParameterRegistry.Get<int>("total_classifications");
			int correctClassifications = ParameterRegistry.Get<int>("correct_classifications");
			double lowerThreshold = ParameterRegistry.Get<double>("lower_threshold");
			double upperThreshold = ParameterRegistry.Get<double>("upper_threshold");

			for (int i = 0; i < predictions.Shape[0]; i++)
			{
				double value = predictions.GetValue<double>(i, 0);
				int target = targets.GetValue<int>(i, 0);

				if (value < lowerThreshold && target == 0 || value > upperThreshold && target == 1)
				{
					correctClassifications++;
				}
			}

			totalClassifications += (int)predictions.Shape[0];

			ParameterRegistry["total_classifications"] = totalClassifications;
			ParameterRegistry["correct_classifications"] = correctClassifications;
		}

		/// <summary>
		/// End a validation scoring session.
		/// Write out results here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreEnd(IRegistry registry, IRegistryResolver resolver)
		{
			string resultKey = ParameterRegistry.Get<string>("result_key");
			double accuracy = (double)ParameterRegistry.Get<int>("correct_classifications") / ParameterRegistry.Get<int>("total_classifications");

			resolver.ResolveSet(resultKey, accuracy, true);
		}
	}
}
