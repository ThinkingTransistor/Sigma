/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Parameterisation.Types
{
	/// <summary>
	/// A ranged parameter type for a certain generic numeric type.
	/// </summary>
	public class RangedParameterType<T> : IParameterType where T : IComparable
	{
		/// <summary>
		/// The minimum value of this parameter type.
		/// </summary>
		public T MinValue { get; }

		/// <summary>
		/// The maximum value of this parameter type.
		/// </summary>
		public T MaxValue { get; }

		/// <summary>
		/// The step size of this parameter type.
		/// </summary>
		public T StepSize { get; }

		/// <summary>
		/// Create a ranged parameter type with a certain minimum and maximum value and a certain step size.
		/// </summary>
		/// <param name="minValue">The minimum value.</param>
		/// <param name="maxValue">The maximum value.</param>
		/// <param name="stepSize">The step size.</param>
		public RangedParameterType(T minValue, T maxValue, T stepSize)
		{
			MinValue = minValue;
			MaxValue = maxValue;
			StepSize = stepSize;
		}
	}
}
