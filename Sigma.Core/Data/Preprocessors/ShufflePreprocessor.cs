using System;
using ManagedCuda.VectorTypes;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors
{
    [Serializable]
    public class ShufflePreprocessor : BasePreprocessor
    {
        /// <summary>
        /// The dimension along which should be shuffled.
        /// Note: Must be >= 0.
        /// </summary>
        public int AlongDimension
        {
            get { return _alongDimension; }
            set
            {
                if (value < 0) throw new ArgumentException($"Along dimension must be >= 0 but given value was {value}.");

                _alongDimension = value;
            }
        }

        /// <inheritdoc />
        public override bool AffectsDataShape => false;

        private Random _random;
        private int _alongDimension;

        /// <summary>
        /// Create a shuffle preprocessor and optionally specify the dominant dimension (along which should be shuffled, batch dimension (0) by default).
        /// </summary>
        /// <param name="alongDimension"></param>
        public ShufflePreprocessor(int alongDimension = 0)
        {
            AlongDimension = alongDimension;
        }

        /// <summary>
        /// Process a certain ndarray with a certain computation handler.
        /// </summary>
        /// <param name="array">The ndarray to process.</param>
        /// <param name="handler">The computation handler to do the processing with.</param>
        /// <returns>An ndarray with the processed contents of the given array (can be the same or a new one).</returns>
        internal override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
        {
            int recordLength = (int) (array.Length / array.Shape[0]);
            long[] firstBufferIndices = new long[array.Shape.Length];
            long[] secondBufferIndices = new long[array.Shape.Length];

            _random = new Random(31415926); // fixed rng for reproducability 

            for (int i = 0; i < array.Shape[0]; i++)
            {
                int swapIndex = _random.Next((int) array.Shape[0]);

                for (int y = 0; y < recordLength; y++)
                {
                    NDArrayUtils.GetIndices(recordLength * i + y, array.Shape, array.Strides, firstBufferIndices);
                    NDArrayUtils.GetIndices(recordLength * swapIndex + y, array.Shape, array.Strides, secondBufferIndices);

                    double firstValue = array.GetValue<double>(firstBufferIndices);
                    double secondValue = array.GetValue<double>(secondBufferIndices);

                    array.SetValue(secondValue, firstBufferIndices);
                    array.SetValue(firstValue, secondBufferIndices);
                }
            }

            return array;
        }
    }
}
