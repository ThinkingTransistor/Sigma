﻿/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;
using Sigma.Core.Data.Preprocessors.Adaptive;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// A normalising preprocessor, which normalises all values in certain named sections to a given range of values.
	/// </summary>
	[Serializable]
	public class NormalisingPreprocessor : BasePreprocessor
	{
		/// <inheritdoc />
		public override bool AffectsDataShape => false;

		public double MinInputValue { get; set; }
		public double MaxInputValue { get; set; }
		public double MinOutputValue { get; set; }
		public double MaxOutputValue { get; set; }

		/// <summary>
		/// Create a normalising preprocessor with a certain input and an output range of [0, 1] and optionally specify for which sections this processor should be applied.
		/// </summary>
		/// <param name="minInputValue">The minimum input value (inclusive).</param>
		/// <param name="maxInputValue">The maximum input value (inclusive).</param>
		/// <param name="sectionNames">The optional section names. If specified, only the given sections are processed.</param>
		public NormalisingPreprocessor(double minInputValue, double maxInputValue, params string[] sectionNames) : this(minInputValue, maxInputValue, 0, 1, sectionNames)
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
		public NormalisingPreprocessor(double minInputValue, double maxInputValue, double minOutputValue, double maxOutputValue, params string[] sectionNames) : base(sectionNames)
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
		}

	    /// <summary>
	    /// Get an adaptive version of this normalising preprocessor.
	    /// </summary>
	    /// <param name="adaptionRate">The optional adaption rate.</param>
	    /// <returns>An adaptive version of this normalising preprocessor.</returns>
	    public IRecordPreprocessor Adaptive(AdaptionRate adaptionRate = AdaptionRate.Every)
	    {
	        return Preprocess(new AdaptiveNormalisingPreprocessor(this, adaptionRate));
	    }

		internal override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			double inputRange = MaxInputValue - MinInputValue;
			double outputRange = MaxOutputValue - MinOutputValue;
			double outputScale = outputRange / inputRange;

			array = handler.Subtract(array, MinInputValue);
			array = handler.Multiply(array, outputScale);
			array = handler.Add(array, MinOutputValue);

			return array;
		}
	}
}
