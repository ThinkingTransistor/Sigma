using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Handlers;
using Sigma.Core.Math;

namespace Sigma.Core.Data.Preprocessors
{
	public class NormalisingPreprocessor : BasePreprocessor
	{
		public int MinInputValue { get; private set; }
		public int MaxInputValue { get; private set; }
		public int MinOutputValue { get; private set; }
		public int MaxOutputValue { get; private set; }

		private int inputRange;
		private int outputRange;
		private int outputOffset;
		private double outputScale;

		public NormalisingPreprocessor(int minInputValue, int maxInputValue) : this(minInputValue, maxInputValue, 0, 1)
		{
		}

		public NormalisingPreprocessor(int minInputValue, int maxInputValue, int minOutputValue, int maxOutputValue)
		{
			if (minInputValue >= maxInputValue)
			{
				throw new ArgumentException($"Minimum input value must be < maximum input value, but minimum input value was {minInputValue} and maximum input value {maxInputValue}.");
			}

			if (minOutputValue >= maxOutputValue)
			{
				throw new ArgumentException($"Minimum output value must be < maximum output value, but minimum output value was {minOutputValue} and maximum output value {maxOutputValue}.");
			}

			this.MinInputValue = minInputValue;
			this.MaxInputValue = maxInputValue;
			this.MinOutputValue = minOutputValue;
			this.MaxOutputValue = maxOutputValue;

			this.inputRange = maxInputValue - minInputValue;
			this.outputRange = maxOutputValue - minOutputValue;
			this.outputOffset = minOutputValue - minInputValue;
			this.outputScale = ((double) outputRange) / inputRange;
		}

		protected override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			handler.Subtract(array, MinInputValue, array);
			handler.Multiply(array, outputScale, array);
			handler.Add(array, outputOffset, array);

			return array;
		}
	}
}
