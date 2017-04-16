/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.ComponentModel;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors.Adaptive
{
    /// <summary>
    /// An adaptive preprocessor that adapts underlying preprocessors to inputs on a certain adaption rate.
    /// </summary>
    /// <typeparam name="TPreprocessor"></typeparam>
    [Serializable]
    public abstract class BaseAdaptivePreprocessor<TPreprocessor> : BasePreprocessor where TPreprocessor : BasePreprocessor
    {
        /// <summary>
        /// The adaption rate to use.
        /// </summary>
        protected AdaptionRate AdaptionRate { get; }

        private readonly TPreprocessor _underlyingPreprocessor;
        private bool _initialAdaptionComplete;

        /// <summary>
        /// Create a base adaptive preprocessor with a certain underlying preprocessor (that will be adapted).
        /// </summary>
        /// <param name="underlyingPreprocessor">The underlying preprocessor.</param>
        /// <param name="sectionNames">The section names to process in this preprocessor (all if null or empty).</param>
        protected BaseAdaptivePreprocessor(TPreprocessor underlyingPreprocessor, params string[] sectionNames) : this(underlyingPreprocessor, AdaptionRate.Initial, sectionNames)
        {
        }

        /// <summary>
        /// Create a base adaptive preprocessor with a certain underlying preprocessor (that will be adapted).
        /// </summary>
        /// <param name="underlyingPreprocessor">The underlying preprocessor.</param>
        /// <param name="adaptionRate">The adaption rate.</param>
        /// <param name="sectionNames">The section names to process in this preprocessor (all if null or empty).</param>
        protected BaseAdaptivePreprocessor(TPreprocessor underlyingPreprocessor, AdaptionRate adaptionRate, params string[] sectionNames) : base(sectionNames)
        {
            if (underlyingPreprocessor == null) throw new ArgumentNullException(nameof(underlyingPreprocessor));
            if (!Enum.IsDefined(typeof(AdaptionRate), adaptionRate)) throw new InvalidEnumArgumentException(nameof(adaptionRate), (int) adaptionRate, typeof(AdaptionRate));


            _underlyingPreprocessor = underlyingPreprocessor;
            AdaptionRate = adaptionRate;
        }

        /// <inheritdoc />
        internal override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
        {
            if (AdaptionRate == AdaptionRate.Every || !_initialAdaptionComplete && AdaptionRate == AdaptionRate.Initial)
            {
                AdaptUnderlyingPreprocessor(_underlyingPreprocessor, array, handler);

                _initialAdaptionComplete = true;
            }

            return _underlyingPreprocessor.ProcessDirect(array, handler);
        }

        /// <summary>
        /// Adapt the underlying preprocessor to the given array using a certain computation handler.
        /// </summary>
        /// <param name="preprocessor">The underlying preprocessor to adapt to the array.</param>
        /// <param name="array">The array.</param>
        /// <param name="handler">The computation handler.</param>
        protected abstract void AdaptUnderlyingPreprocessor(TPreprocessor preprocessor, INDArray array, IComputationHandler handler);
    }

    /// <summary>
    /// The adaption
    /// </summary>
    public enum AdaptionRate
    {
        /// <summary>
        /// Only call AdaptUnderlyingPreprocessor once.
        /// </summary>
        Initial,

        /// <summary>
        /// Call AdaptUnderlyingPreprocessor every time anything is processed.
        /// </summary>
        Every
    }
}
