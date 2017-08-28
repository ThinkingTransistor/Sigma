/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using log4net;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Processors
{
	/// <summary>
	/// A hook that attempts to find an input that causes the network to output targets close to the desired targets.
	/// Note: Currently only works with single-input / single-output models.
	/// </summary>
	[Serializable]
	public class TargetMaximisationHook : BaseHook
	{
		[NonSerialized]
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create a hook with a certain time step and a set of required global registry entries. 
		/// </summary>
		/// <param name="timestep">The time step.</param>
		/// <param name="desiredTargets">The targets for which the inputs should be optimised for.</param>
		/// <param name="desiredCost">The maximum factor of difference between approached targets and desired targets (squared difference).</param>
		/// <param name="sharedResultInputKey">The shared key under which the result input will be available.</param>
		/// <param name="sharedResultSuccessKey">The shared key under which the result success flag will be available.</param>
		public TargetMaximisationHook(ITimeStep timestep, INDArray desiredTargets, string sharedResultSuccessKey, string sharedResultInputKey, double desiredCost = 0.06)
			: base(timestep, "network.self", "optimiser.self")
		{
			if (desiredTargets == null) throw new ArgumentNullException(nameof(desiredTargets));
			if (sharedResultInputKey == null) throw new ArgumentNullException(nameof(sharedResultInputKey));
			if (sharedResultSuccessKey == null) throw new ArgumentNullException(nameof(sharedResultSuccessKey));

			ParameterRegistry["desired_targets"] = desiredTargets;
			ParameterRegistry["desired_cost"] = desiredCost;
			ParameterRegistry["max_optimisation_steps"] = 1024;
			ParameterRegistry["max_optimisation_attempts"] = 2;
			ParameterRegistry["shared_result_input_key"] = sharedResultInputKey;
			ParameterRegistry["shared_result_success_key"] = sharedResultSuccessKey;

			InvokePriority = 10000;
		}

		/// <summary>
		/// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
		/// </summary>
		/// <param name="registry">The registry containing the required values for this hook's execution.</param>
		/// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
		public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
		{
			// we need copies of network and optimiser as to not affect the current internal state
			INetwork network = (INetwork)resolver.ResolveGetSingle<INetwork>("network.self").DeepCopy();
			BaseGradientOptimiser optimiser = (BaseGradientOptimiser) resolver.ResolveGetSingle<BaseGradientOptimiser>("optimiser.self").ShallowCopy(); 
			INDArray desiredTargets = ParameterRegistry.Get<INDArray>("desired_targets");
			IComputationHandler handler = new DebugHandler(Operator.Handler);

			long[] inputShape = network.YieldExternalInputsLayerBuffers().First().Parameters.Get<long[]>("shape");

			IDictionary<string, INDArray> block = DataUtils.MakeBlock("targets", desiredTargets); // desired targets don't change during execution

			double desiredCost = ParameterRegistry.Get<double>("desired_cost"), currentCost = Double.MaxValue;
			int maxOptimisationAttempts = ParameterRegistry.Get<int>("max_optimisation_attempts");
			int maxOptimisationSteps = ParameterRegistry.Get<int>("max_optimisation_steps");
			int optimisationSteps = 0;
			INDArray maximisedInputs = CreateRandomisedInput(handler, inputShape); 

			for (int i = 0; i < maxOptimisationAttempts; i++)
			{
				optimisationSteps = 0;

				do
				{
					// trace current inputs and run network as normal
					uint traceTag = handler.BeginTrace();
					block["inputs"] = handler.Trace(maximisedInputs.Reshape(ArrayUtils.Concatenate(new[] {1L, 1L}, inputShape)), traceTag);

					DataUtils.ProvideExternalInputData(network, block);
					network.Run(handler, trainingPass: false);

					// fetch current outputs and optimise against them (towards desired targets)
					INDArray currentTargets = network.YieldExternalOutputsLayerBuffers().First(buffer => buffer.ExternalOutputs.Contains("external_default"))
						.Outputs["external_default"].Get<INDArray>("activations");
					currentTargets = handler.RowWise(handler.FlattenTimeAndFeatures(currentTargets), handler.SoftMax);

					//INDArray logPredictions = handler.Log(currentTargets);
					//INDArray a = handler.Multiply(desiredTargets, logPredictions);

					//INDArray inverseTargets = handler.Subtract(1, desiredTargets);
					//INDArray inversePredictions = handler.Subtract(1 + 1e-6, currentTargets);
					//INDArray b = handler.Multiply(inverseTargets, handler.Log(inversePredictions));

					INumber cost = handler.Sum(handler.Pow(handler.Subtract(currentTargets, desiredTargets), 2));
					//INumber cost = handler.Divide(handler.Sum(handler.Add(a, b)), -currentTargets.Shape[0]); // TODO change to softmax ce cost, not sure why it doesn't work right now
																											   // (it's even using the same optimiser?)

					handler.ComputeDerivativesTo(cost);

					INDArray gradient = handler.GetDerivative(block["inputs"]);
					maximisedInputs = handler.ClearTrace(optimiser.Optimise("inputs", block["inputs"], gradient, handler));

					currentCost = cost.GetValueAs<double>();

					if (currentCost <= desiredCost)
					{
						goto Validation;
					}

				} while (++optimisationSteps < maxOptimisationSteps);

				maximisedInputs = CreateRandomisedInput(handler, inputShape); // reset input
			}

			Validation:
			maximisedInputs.ReshapeSelf(inputShape);

			string sharedResultInput = ParameterRegistry.Get<string>("shared_result_input_key");
			string sharedResultSuccess = ParameterRegistry.Get<string>("shared_result_success_key");

			if (optimisationSteps >= maxOptimisationSteps)
			{
				_logger.Debug($"Aborted target maximisation for {desiredTargets}, failed after {maxOptimisationSteps} optimisation steps in {maxOptimisationAttempts} attempts (exceeded limit, current cost {currentCost} but desired {desiredCost}).");

				resolver.ResolveSet(sharedResultSuccess, false, addIdentifierIfNotExists: true);
			}
			else
			{
				_logger.Debug($"Successfully finished target optimisation for {desiredTargets} after {optimiser} optimisation steps.");

				resolver.ResolveSet(sharedResultSuccess, true, addIdentifierIfNotExists: true);
				resolver.ResolveSet(sharedResultInput, maximisedInputs, addIdentifierIfNotExists: true);
			}
		}

		private INDArray CreateRandomisedInput(IComputationHandler handler, long[] shape)
		{
			INDArray randomInput = handler.NDArray(shape);

			new GaussianInitialiser(standardDeviation: 0.01).Initialise(randomInput, handler, new Random()); // new random because this shouldn't affect the global rng seed

			return randomInput;
		}
	}
}
