/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Scorers
{
	public class ValidationAccuracyScorer : BaseValidationScorer
	{
		///  <summary>
		///  Create a validation accuracy scorer hook for a certain validation iterator.
		///  Note:	The "external_default" output may not always be the actual final output in your model.
		/// 		If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		///  </summary>
		///  <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="resultBaseEntry">The base entry under which the results will be available (base entry + tops[i]).</param>
		/// <param name="tops">The tops that should be scored (e.g. top 1, top 3, top5).</param>
		/// <param name="timestep">The time step.</param>
		public ValidationAccuracyScorer(string validationIteratorName, string resultBaseEntry, ITimeStep timestep, params int[] tops) : base(validationIteratorName, timestep)
		{
			if (tops == null) throw new ArgumentNullException(nameof(tops));
			if (tops.Length == 0) throw new ArgumentException($"The tops must be of length > 0 (otherwise what should be scored? It doesn't make sense).");

			ParameterRegistry["tops"] = tops;
			ParameterRegistry["result_base_entry"] = resultBaseEntry;
		}

		/// <summary>
		/// Begin a validation scoring session.
		/// Reset the scoring here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreBegin(IRegistry registry, IRegistryResolver resolver)
		{
			int[] tops = ParameterRegistry.Get<int[]>("tops");

			foreach (int t in tops)
			{
				ParameterRegistry[$"correct_classifications_top{t}"] = 0;
			}

			ParameterRegistry["total_classifications"] = 0;
		}

		/// <summary>
		/// Score an intermediate result (part of the entire validation dataset was used as specified by the validation iterator).
		/// </summary>
		/// <param name="predictions">The predictions.</param>
		/// <param name="targets">The targets.</param>
		/// <param name="handler">The computation handler.</param>
		protected override void ScoreIntermediate(INDArray predictions, INDArray targets, IComputationHandler handler)
		{
			int[] tops = ParameterRegistry.Get<int[]>("tops");

			predictions = handler.RowWise(handler.FlattenTimeAndFeatures(predictions), handler.SoftMax);
			var perRowTopPredictions = handler.RowWiseTransform(predictions, 
				row => row.GetDataAs<double>().Data.Select((x, i) => new KeyValuePair<double, int>(x, i)).OrderByDescending(x => x.Key).Select(p => p.Value).ToArray()).ToList();

			int[] targetIndices = handler.RowWiseTransform(handler.FlattenTimeAndFeatures(targets), handler.MaxIndex);

			foreach (int top in tops)
			{
				int correctClassifications = perRowTopPredictions.Where((rowPredictions, rowIndex) => rowPredictions.Take(top).Any(predicted => predicted == targetIndices[rowIndex])).Count();

				ParameterRegistry[$"correct_classifications_top{top}"] = ParameterRegistry.Get<int>($"correct_classifications_top{top}") + correctClassifications;
			}
			
			ParameterRegistry["total_classifications"] = ParameterRegistry.Get<int>("total_classifications") + targetIndices.Length;
		}

		/// <summary>
		/// End a validation scoring session.
		/// Write out results here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreEnd(IRegistry registry, IRegistryResolver resolver)
		{
			int[] tops = ParameterRegistry.Get<int[]>("tops");

			foreach (int top in tops)
			{
				string resultBaseEntry = ParameterRegistry.Get<string>("result_base_entry");

				int totalClassifications = ParameterRegistry.Get<int>("total_classifications");
				int correctClassifications = ParameterRegistry.Get<int>($"correct_classifications_top{top}");

				double score = ((double) correctClassifications) / totalClassifications;

				resolver.ResolveSet(resultBaseEntry + top, score, addIdentifierIfNotExists: true);
			}
		}
	}
}
