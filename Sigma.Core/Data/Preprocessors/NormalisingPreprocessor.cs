/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// A normalising preprocessor, which normalises all values in certain named sections to a given range of values.
	/// </summary>
	public class NormalisingPreprocessor : BasePreprocessor
	{
		public override bool AffectsDataShape => false;

		public int MinInputValue { get; }
		public int MaxInputValue { get; private set; }
		public int MinOutputValue { get; }
		public int MaxOutputValue { get; private set; }

		private readonly double _outputScale;

		/// <summary>
		/// Create a normalising preprocessor with a certain input and an output range of [0, 1] and optionally specify for which sections this processor should be applied.
		/// </summary>
		/// <param name="minInputValue">The minimum input value (inclusive).</param>
		/// <param name="maxInputValue">The maximum input value (inclusive).</param>
		/// <param name="sectionNames">The optional section names. If specified, only the given sections are processed.</param>
		public NormalisingPreprocessor(int minInputValue, int maxInputValue, params string[] sectionNames) : this(minInputValue, maxInputValue, 0, 1, sectionNames)
		{
		}

		/// <summary>
		/// Create a normalising preprocessor with a certain input and output range and optionally specify for which sections this processor should be applied.
		/// </summary>
		/// <param name="minInputValue">The minimum input value (inclusive).</param>
		/// <param name="maxInputValue">The maximum input value (inclusive).</param>
		/// <param name="minOutputValue">The minimum output value (inclusive).</param>
		/// <param name="maxOutputValue">The maximum output value (inclusive).</param>
		/// <param name="sectionNames">The optional section names. If specified, only the given sections are processed.</param>
		public NormalisingPreprocessor(int minInputValue, int maxInputValue, int minOutputValue, int maxOutputValue, params string[] sectionNames) : base(sectionNames)
		{
			if (minInputValue >= maxInputValue)
			{
				throw new ArgumentException($"Minimum input value must be < maximum input value, but minimum input value was {minInputValue} and maximum input value {maxInputValue}.");
			}

			if (minOutputValue >= maxOutputValue)
			{
				throw new ArgumentException($"Minimum output value must be < maximum output value, but minimum output value was {minOutputValue} and maximum output value {maxOutputValue}.");
			}

			MinInputValue = minInputValue;
			MaxInputValue = maxInputValue;
			MinOutputValue = minOutputValue;
			MaxOutputValue = maxOutputValue;

			int inputRange = maxInputValue - minInputValue;
			int outputRange = maxOutputValue - minOutputValue;
			_outputScale = ((double) outputRange) / inputRange;
		}

		protected override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			array = handler.Subtract(array, MinInputValue);
			array = handler.Multiply(array, _outputScale);
			array = handler.Add(array, MinOutputValue);

			return array;
		}
	}
}
