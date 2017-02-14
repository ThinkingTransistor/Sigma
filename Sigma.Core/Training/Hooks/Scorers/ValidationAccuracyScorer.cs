/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

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
		/// 			If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		///  </summary>
		///  <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="resultEntry"></param>
		/// <param name="timestep">The time step.</param>
		public ValidationAccuracyScorer(string validationIteratorName, string resultEntry, ITimeStep timestep) : base(validationIteratorName, timestep)
		{
			ParameterRegistry["result_entry"] = resultEntry;
		}

		/// <summary>
		/// Begin a validation scoring session.
		/// Reset the scoring here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreBegin(IRegistry registry, IRegistryResolver resolver)
		{
			ParameterRegistry["correct_classifications_top1"] = 0;
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
			predictions = handler.RowWise(handler.FlattenTimeAndFeatures(predictions), handler.SoftMax);
			targets = handler.FlattenTimeAndFeatures(targets); // TODO add safeguard against calling RowWise on non-flattened ndarrays

			int[] predictedIndices = handler.RowWiseTransform(predictions, handler.MaxIndex);
			int[] targetIndices = handler.RowWiseTransform(targets, handler.MaxIndex);

			int correctClassifications = 0;

			for (int i = 0; i < predictedIndices.Length; i++)
			{
				if (predictedIndices[i] == targetIndices[i])
				{
					correctClassifications++;
				}
			}

			ParameterRegistry["correct_classifications_top1"] = ParameterRegistry.Get<int>("correct_classifications_top1") + correctClassifications;
			ParameterRegistry["total_classifications"] = ParameterRegistry.Get<int>("total_classifications") + predictedIndices.Length;
		}

		/// <summary>
		/// End a validation scoring session.
		/// Write out results here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected override void ScoreEnd(IRegistry registry, IRegistryResolver resolver)
		{
			string resultEntry = ParameterRegistry.Get<string>("result_entry");

			int totalClassifications = ParameterRegistry.Get<int>("total_classifications");
			int correctClassificationsTop1 = ParameterRegistry.Get<int>("correct_classifications_top1");

			double score = ((double) correctClassificationsTop1) / totalClassifications;

			resolver.ResolveSet(resultEntry, score, addIdentifierIfNotExists: true);
		}
	}
}
