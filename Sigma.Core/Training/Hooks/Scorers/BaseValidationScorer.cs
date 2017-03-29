/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Architecture;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Handlers;
using Sigma.Core.Layers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;

namespace Sigma.Core.Training.Hooks.Scorers
{
	/// <summary>
	/// A base implementation for validation scorers that takes care of automatically getting the final output of a and staging the scoring process.
	/// </summary>
	[Serializable]
	public abstract class BaseValidationScorer : BaseHook
	{
		/// <summary>
		/// Create a validation scorer hook for a certain validation iterator.
		/// Note:	The "external_default" output may not always be the actual final output in your model.
		///			If your model contains multiple output you may need to explicitly specify the actual final output with this alias.
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="timestep">The time step.</param>
		protected BaseValidationScorer(string validationIteratorName, ITimeStep timestep) : this(validationIteratorName, "external_default", timestep)
		{
		}

		/// <summary>
		/// Create a validation scorer hook for a certain validation iterator.
		/// </summary>
		/// <param name="validationIteratorName">The validation data iterator name (as in the trainer).</param>
		/// <param name="finalExternalOutputAlias">The final external output alias (where the actual output is).</param>
		/// <param name="timestep">The time step.</param>
		protected BaseValidationScorer(string validationIteratorName, string finalExternalOutputAlias, ITimeStep timestep) : base(timestep, "network.self", "trainer.self")
		{
			if (validationIteratorName == null) throw new ArgumentNullException(nameof(validationIteratorName));
			if (finalExternalOutputAlias == null) throw new ArgumentNullException(nameof(finalExternalOutputAlias));

			DefaultTargetMode = TargetMode.Global;

			ParameterRegistry["validation_iterator_name"] = validationIteratorName;
			ParameterRegistry["final_external_output_alias"] = finalExternalOutputAlias;
			ParameterRegistry["output_activations_alias"] = "activations";
			ParameterRegistry["targets_alias"] = "targets";
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			INetwork network = resolver.ResolveGetSingle<INetwork>("network.self");
			ITrainer trainer = resolver.ResolveGetSingle<ITrainer>("trainer.self");
			string validationIteratorName = ParameterRegistry.Get<string>("validation_iterator_name");
			string finalExternalOutputAlias = ParameterRegistry.Get<string>("final_external_output_alias");
			string activationsAlias = ParameterRegistry.Get<string>("output_activations_alias");
			string targetsAlias = ParameterRegistry.Get<string>("targets_alias");

			if (!trainer.AdditionalNameDataIterators.ContainsKey(validationIteratorName))
			{
				throw new InvalidOperationException($"Additional named data iterator for validation with name \"{validationIteratorName}\" does not exist in referenced trainer {trainer} but is required.");
			}

			IDataIterator validationIterator = trainer.AdditionalNameDataIterators[validationIteratorName];

			ScoreBegin(registry, resolver);

			foreach (var block in validationIterator.Yield(Operator.Handler, Operator.Sigma))
			{
				trainer.ProvideExternalInputData(network, block);
				network.Run(Operator.Handler, trainingPass: false);

				INDArray finalOutputPredictions = null;

				foreach (ILayerBuffer layerBuffer in network.YieldExternalOutputsLayerBuffers())
				{
					foreach (string outputAlias in layerBuffer.ExternalOutputs)
					{
						if (outputAlias.Equals(finalExternalOutputAlias))
						{
							finalOutputPredictions = Operator.Handler.ClearTrace(layerBuffer.Outputs[outputAlias].Get<INDArray>(activationsAlias));

							goto FoundOutput;
						}
					};
				}

				throw new InvalidOperationException($"Cannot find final output with alias \"{finalExternalOutputAlias}\" in the current network (but is required to score validation).");

			FoundOutput:
				ScoreIntermediate(finalOutputPredictions, block[targetsAlias], Operator.Handler);		
			}

			ScoreEnd(registry, resolver);
		}

		/// <summary>
		/// Begin a validation scoring session.
		/// Reset the scoring here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected abstract void ScoreBegin(IRegistry registry, IRegistryResolver resolver);

		/// <summary>
		/// Score an intermediate result (part of the entire validation dataset was used as specified by the validation iterator).
		/// </summary>
		/// <param name="predictions">The predictions.</param>
		/// <param name="targets">The targets.</param>
		/// <param name="handler">The computation handler.</param>
		protected abstract void ScoreIntermediate(INDArray predictions, INDArray targets, IComputationHandler handler);

		/// <summary>
		/// End a validation scoring session.
		/// Write out results here.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		protected abstract void ScoreEnd(IRegistry registry, IRegistryResolver resolver);
	}
}
