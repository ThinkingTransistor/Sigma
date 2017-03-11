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
    public class AdadeltaOptimiser : BaseMemoryGradientOptimiser<INDArray>
    {
        /// <summary>
        /// Create a base memory gradient optimiser with an optional external output cost alias to use. 
        /// </summary>
        /// <param name="externalCostAlias">The optional external output identifier by which to detect cost layers (defaults to "external_cost").</param>
        public AdadeltaOptimiser(string memoryIdentifier, string externalCostAlias = "external_cost") : base("memory_previous_param_update", externalCostAlias)
        {
        }

        /// <inheritdoc />
        protected override INDArray Optimise(string paramIdentifier, INDArray parameter, INDArray gradient, IComputationHandler handler)
        {
            //double momentum = Registry.Get<double>("momentum"), smoothing = Registry.Get<double>("smoothing");

            //INDArray previousParamUpdateE = GetMemory(paramIdentifier, () => SquareRootSmoothed(handler.Multiply(gradient, gradient), smoothing, handler));
            //INDArray parameterUpdateE = handler.Add(handler.Multiply(previousParamUpdateE, momentum), handler.Multiply());

            //INDArray gradientRms = SquareRootSmoothed(handler.Multiply(gradient, gradient), smoothing, handler);

            //INDArray parameterUpdate = handler.Multiply(handler.Divide())

            throw new NotImplementedException();
        }

        private INDArray SquareRootSmoothed(INDArray array, double smoothing, IComputationHandler handler)
        {
            return handler.SquareRoot(handler.Add(array, smoothing));
        }

        /// <summary>
        /// Deep copy this object.
        /// </summary>
        /// <returns>A deep copy of this object.</returns>
        public override object DeepCopy()
        {
            throw new NotImplementedException();
        }
    }
}
