/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// A one-hot preprocessor, which encodes a one-dimensional feature dimension in the one-hot format. 
	/// <example>
	///	For example, with a given possible value range of [0, 9]
	///									   4th
	///		an input of 4 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0]		
	///		
	///												7th
	///		an input of 7 becomes [0, 0, 0, 0, 0, 0, 1, 0, 0]
	/// </example>
	/// </summary>
	public class OneHotPreprocessor : BasePreprocessor
	{
		public override bool AffectsDataShape => true;

		private readonly Dictionary<object, int> _valueToIndexMapping;

		/// <summary>
		/// Create a one-hot preprocessor with possible values within a certain integer range for all sections.
		/// </summary>
		/// <param name="minValue">The minimum integer value of all possible values.</param>
		/// <param name="maxValue">The maximum integer value of all possible values.</param>
		public OneHotPreprocessor(int minValue, int maxValue) : this(sectionName: null, minValue: minValue, maxValue: maxValue)
		{
		}

		/// <summary>
		/// Create a one-hot preprocessor with possible values within a certain integer range for a specific section.
		/// </summary>
		/// <param name="sectionName">The specific section this preprocessor should be applied to.</param>
		/// <param name="minValue">The minimum integer value of all possible values.</param>
		/// <param name="maxValue">The maximum integer value of all possible values.</param>
		public OneHotPreprocessor(string sectionName, int minValue, int maxValue) : this(sectionName: sectionName, possibleValues: ArrayUtils.Range(minValue, maxValue).As<int, object>())
		{
		}

		/// <summary>
		/// Create a one-hot preprocessor with an array of possible values and optionally for a specific section.
		/// </summary>
		/// <param name="sectionName">The optional specific this processor should be applied to.</param>
		/// <param name="possibleValues">All possible values that this one-hot preprocessor should encode (have to be known ahead of time).</param>
		public OneHotPreprocessor(string sectionName, params object[] possibleValues) : base(sectionName == null ? null : new[] { sectionName})
		{
			if (possibleValues == null)
			{
				throw new ArgumentNullException(nameof(possibleValues));
			}

			if (possibleValues.Length == 0)
			{
				throw new ArgumentException("Possible values cannot be empty.");
			}

			_valueToIndexMapping = new Dictionary<object, int>();

			for (int i = 0; i < possibleValues.Length; i++)
			{
				if (_valueToIndexMapping.ContainsKey(possibleValues[i]))
				{
					throw new ArgumentException($"Possible values must be unique, but there was a duplicate value {possibleValues[i]} at index {i}.");
				}

				_valueToIndexMapping.Add(possibleValues[i], i);
			}
		}

		protected override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			//BTF with single feature dimension
			if (array.Rank != 3)
			{
				throw new ArgumentException($"Cannot one-hot encode ndarrays which are not of rank 3 (BTF with single feature dimension), but given ndarray was of rank {array.Rank}.");
			}

			if (array.Shape[2] != 1)
			{
				throw new ArgumentException($"Cannot one-hot encode ndarrays whose feature shape (index 2) is not 1, but ndarray.shape[2] was {array.Shape[2]}");
			}

			INDArray encodedArray = handler.NDArray(array.Shape[0], array.Shape[1], _valueToIndexMapping.Count);

			long[] bufferIndices = new long[3];

			for (long i = 0; i < array.Length; i++)
			{
				bufferIndices = NDArrayUtils.GetIndices(i, array.Shape, array.Strides, bufferIndices);

				object value = array.GetValue<int>(bufferIndices);

				if (!_valueToIndexMapping.ContainsKey(value))
				{
					throw new ArgumentException($"Cannot one-hot encode unknown value {value}, value was not registered as a possible value.");
				}

				bufferIndices[2] = _valueToIndexMapping[value];

				encodedArray.SetValue(1, bufferIndices);
			}

			return encodedArray;
		}
	}
}
