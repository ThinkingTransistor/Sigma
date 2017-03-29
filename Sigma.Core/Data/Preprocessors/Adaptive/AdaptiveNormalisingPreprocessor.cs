/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors.Adaptive
{
    /// <summary>
    /// An adaptive normalising preprocessor, which adapts the normalised input range to every processed block.
    /// </summary>
    [Serializable]
    public class AdaptiveNormalisingPreprocessor : BaseAdaptivePreprocessor<NormalisingPreprocessor>
    {
        /// <inheritdoc />
        public override bool AffectsDataShape => false;

        /// <summary>
        /// Create an adaptive normalising preprocessor with a certain underlying preprocessor.
        /// </summary>
        /// <param name="underlyingPreprocessor">The underlying preprocessor.</param>
        /// <param name="adaptionRate">The adaption rate.</param>
        public AdaptiveNormalisingPreprocessor(NormalisingPreprocessor underlyingPreprocessor, AdaptionRate adaptionRate = AdaptionRate.Every) : base(underlyingPreprocessor, adaptionRate)
        {
        }

        /// <summary>
        /// Create an adaptive normalising preprocessor with a certain output range.
        /// </summary>
        /// <param name="minOutputValue">The min output value.</param>
        /// <param name="maxOutputValue">The max output value.</param>
        /// <param name="adaptionRate">The optional adaption rate.</param>
        public AdaptiveNormalisingPreprocessor(double minOutputValue, double maxOutputValue, AdaptionRate adaptionRate = AdaptionRate.Every) : this(new NormalisingPreprocessor(Double.NegativeInfinity, Double.PositiveInfinity, minOutputValue, maxOutputValue), adaptionRate)
        {
        }

        /// <summary>
        /// Adapt the underlying preprocessor to the given array using a certain computation handler.
        /// </summary>
        /// <param name="preprocessor">The underlying preprocessor to adapt to the array.</param>
        /// <param name="array">The array.</param>
        /// <param name="handler">The computation handler.</param>
        protected override void AdaptUnderlyingPreprocessor(NormalisingPreprocessor preprocessor, INDArray array, IComputationHandler handler)
        {
            preprocessor.MinInputValue = handler.Min(array).GetValueAs<int>();
            preprocessor.MaxInputValue = handler.Max(array).GetValueAs<int>();
        }
    }
}
