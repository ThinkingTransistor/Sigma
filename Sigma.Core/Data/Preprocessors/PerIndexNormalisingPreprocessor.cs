/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// A normalising preprocessor that operators per feature index.
	/// </summary>
	[Serializable]
	public class PerIndexNormalisingPreprocessor : BasePreprocessor
	{
		public override bool AffectsDataShape => false;

		public double MinOutputValue { get; }
		public double MaxOutputValue { get; private set; }

		private readonly IDictionary<int, double[]> _perIndexMinMaxMappings;

		/// <summary>
		/// Create a per index normalising preprocessor with a certain output range and index mappings (order is index, min value, max value).
		/// </summary>
		/// <param name="minOutputValue">The minimum output value.</param>
		/// <param name="maxOutputValue">The maximum output value.</param>
		/// <param name="sectionName">The section name to pre-process.</param>
		/// <param name="perIndexMinMaxMappings">The per index min max value mappings.</param>
		public PerIndexNormalisingPreprocessor(double minOutputValue, double maxOutputValue, string sectionName, params object[] perIndexMinMaxMappings) : this(minOutputValue, maxOutputValue, sectionName, ToPerIndexMinMaxMappings(perIndexMinMaxMappings))
		{
		}

		public PerIndexNormalisingPreprocessor(double minOutputValue, double maxOutputValue, string sectionName, IDictionary<int, double[]> perIndexMinMaxMappings) : base(new[] { sectionName })
		{
			if (perIndexMinMaxMappings == null) throw new ArgumentNullException(nameof(perIndexMinMaxMappings));

			MinOutputValue = minOutputValue;
			MaxOutputValue = maxOutputValue;
			_perIndexMinMaxMappings = perIndexMinMaxMappings;
		}

		protected override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			double outputRange = MaxOutputValue - MinOutputValue;

			for (int i = 0; i < array.Shape[0]; i++)
			{
				for (int j = 0; j < array.Shape[1]; j++)
				{
					foreach (int index in _perIndexMinMaxMappings.Keys)
					{
						double[] minMaxRange = _perIndexMinMaxMappings[index];
						double value = array.GetValue<double>(i, j, index);

						value = (value - minMaxRange[0]) / minMaxRange[2];
						value = (value + MinOutputValue) * outputRange;

						array.SetValue(value, i, j, index);
					}
				}
			}

			return array;
		}

		private static IDictionary<int, double[]> ToPerIndexMinMaxMappings(object[] args)
		{
			if (args.Length % 3 != 0)
			{
				throw new ArgumentException($"Invalid per index mapping arguments, number of arguments must be multiple of 3 but was {args.Length} (index, min, max).");
			}

			IDictionary<int, double[]> perIndexMinMaxMappings = new Dictionary<int, double[]>();

			for (var i = 0; i < args.Length; i += 3)
			{
				int index = GetArgAtIndexAs<int>(args, i);
				double min = GetArgAtIndexAs<double>(args, i + 1);
				double max = GetArgAtIndexAs<double>(args, i + 2);

				if (perIndexMinMaxMappings.ContainsKey(index))
				{
					throw new ArgumentException($"Duplicate per index mapping for index {index} at argument index {i}.");
				}

				perIndexMinMaxMappings.Add(index, new[] { min, max, max - min });
			}

			return perIndexMinMaxMappings;
		}

		private static T GetArgAtIndexAs<T>(object[] args, int index)
		{
			if (!(args[index] is T))
			{
				throw new ArgumentException($"Argument at index {index} should be of type {typeof(T)} but was {args[index]}.");
			}

			return (T) args[index];
		}
	}
}
