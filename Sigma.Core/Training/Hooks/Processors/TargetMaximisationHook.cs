/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Architecture;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Utils;

namespace Sigma.Core.Training.Hooks.Processors
{
    public class TargetMaximisationHook : BaseHook
    {
        /// <summary>
        /// Create a hook with a certain time step and a set of required global registry entries. 
        /// </summary>
        /// <param name="timestep">The time step.</param>
        /// <param name="desiredTargets">The targets for which the inputs should be optimised for.</param>
        /// <param name="desiredCost">The maximum factor of difference between approached targets and desired targets (squared difference).</param>
        public TargetMaximisationHook(ITimeStep timestep, INDArray desiredTargets, double desiredCost = 0.5) : base(timestep, "network.self", "optimiser.self")
        {
            if (desiredTargets == null) throw new ArgumentNullException(nameof(desiredTargets));

            ParameterRegistry["desired_targets"] = desiredTargets;
            ParameterRegistry["desired_cost"] = desiredCost;
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
            BaseGradientOptimiser optimiser = (BaseGradientOptimiser) resolver.ResolveGetSingle<BaseGradientOptimiser>("optimiser.self").DeepCopy();
            INDArray desiredTargets = ParameterRegistry.Get<INDArray>("desired_targets");
            IComputationHandler handler = Operator.Handler;

            long[] inputShape = network.YieldExternalInputsLayerBuffers().First().Parameters.Get<long[]>("shape");
            INDArray maximisedInputs = CreateRandomisedInput(handler, ArrayUtils.Concatenate(new[] { 1L, 1L }, inputShape)); // BTF dimension ordering
            IDictionary<string, INDArray> block = new Dictionary<string, INDArray>();

            block["targets"] = desiredTargets; // desired targets don't change during execution

            double desiredCost = ParameterRegistry.Get<double>("desired_cost"), currentCost;

            do
            {
                uint traceTag = handler.BeginTrace();
                block["inputs"] = handler.Trace(maximisedInputs, traceTag);

                DataProviderUtils.ProvideExternalInputData(network, block);
                network.Run(handler, trainingPass: false);

                INDArray currentTargets = network.YieldExternalOutputsLayerBuffers().First(b => b.ExternalOutputs.Contains("external_default"))
                    .Outputs["external_default"].Get<INDArray>("activations");
                INumber squaredDifference = handler.Sum(handler.Pow(handler.Subtract(handler.SoftMax(currentTargets), desiredTargets), 2));

                handler.ComputeDerivativesTo(squaredDifference);

                INDArray gradient = handler.GetDerivative(block["inputs"]);
                maximisedInputs = optimiser.Optimise("inputs", block["inputs"], gradient, handler);

                currentCost = squaredDifference.GetValueAs<double>();
            } while (currentCost > desiredCost);
        }

        private INDArray CreateRandomisedInput(IComputationHandler handler, long[] shape)
        {
            INDArray randomInput = handler.NDArray(shape);

            new GaussianInitialiser(standardDeviation: 0.1).Initialise(randomInput, handler, new Random()); // new random because this shouldn't affect the global rng seed

            return randomInput;
        }
    }
}
