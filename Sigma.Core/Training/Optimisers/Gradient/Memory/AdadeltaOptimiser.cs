/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Optimisers.Gradient.Memory
{
    /// <summary>
    /// An adadelta optimiser, which optimises parameters using second-order information of past gradients and parameter updates. 
    /// </summary>
    [Serializable]
    public class AdadeltaOptimiser : BaseMemoryGradientOptimiser<INDArray>
    {
        /// <summary>
        /// Create an adadelta optimiser with a certain decay rate (and optionally a smoothing constant).
        /// </summary>
        /// <param name="decayRate">The decay rate.</param>
        /// <param name="smoothing">The optional smoothing constant.</param>
        /// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
        public AdadeltaOptimiser(double decayRate, double smoothing = 1E-6, string externalCostAlias = "external_cost") : base("memory_previous_update_gradient", externalCostAlias)
        {
            Registry.Set("decay_rate", decayRate, typeof(double));
            Registry.Set("smoothing", smoothing, typeof(double));
        }

        /// <inheritdoc />
        protected override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
        {
            // implementation according to the reference algorithm 1 in the published paper "ADADELTA: AN ADAPTIVE LEARNING RATE METHOD"
            double decayRate = Registry.Get<double>("decay_rate"), smoothing = Registry.Get<double>("smoothing");
            string memoryIdentifierUpdate = paramIdentifier + "_update", memoryIdentifierGradient = paramIdentifier + "_gradient";

            // get accumulated gradients / update if memorised, otherwise initialise empty (all zeroes)
            INDArray previousAccumulatedGradient = GetMemory(memoryIdentifierGradient, () => handler.NDArray((long[]) gradient.Shape.Clone()));
            INDArray previousAccumulatedUpdate = GetMemory(memoryIdentifierUpdate, () => handler.NDArray((long[]) gradient.Shape.Clone()));

            // compute accumulated decayed gradients
            INDArray currentGradientDecayed = handler.Multiply(handler.Multiply(gradient, gradient), 1.0 - decayRate);
            INDArray currentAccumulatedGradient = handler.Add(handler.Multiply(previousAccumulatedGradient, decayRate), currentGradientDecayed);

            // compute previous accumulated gradient root mean squared (rms) and previous accumulated update rms
            INDArray previousUpdateRms = SquareRootSmoothed(previousAccumulatedUpdate, smoothing, handler);
            INDArray gradientRms = SquareRootSmoothed(currentAccumulatedGradient, smoothing, handler);

            // compute parameter update using previous accumulated gradient / update rms
            INDArray update = handler.Multiply(handler.Multiply(handler.Divide(previousUpdateRms, gradientRms), gradient), -1.0);

            // compute current accumulated squared decayed updates for next iteration
            INDArray squaredUpdateDecayed = handler.Multiply(handler.Multiply(update, update), 1.0 - decayRate);
            INDArray currentAccumulatedUpdate = handler.Add(handler.Multiply(previousAccumulatedUpdate, decayRate), squaredUpdateDecayed);

            // store accumulated values for next iteration
            SetMemory(memoryIdentifierGradient, currentAccumulatedGradient);
            SetMemory(memoryIdentifierUpdate, currentAccumulatedUpdate);

            ExposeParameterUpdate(paramIdentifier, update);

            // compute optimised parameter using computed update
            return handler.Add(parameter, update);
        }

        private INDArray SquareRootSmoothed(INDArray array, double smoothing, IComputationHandler handler)
        {
            return handler.SquareRoot(handler.Add(array, smoothing));
        }

        /// <inheritdoc />
        public override object DeepCopy()
        {
            return new AdadeltaOptimiser(Registry.Get<double>("decay_rate"), Registry.Get<double>("smoothing"), ExternalCostAlias);
        }
    }
}
