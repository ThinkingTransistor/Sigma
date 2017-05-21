/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors.Adaptive
{
	/// <summary>
	/// An adaptive version of the <see cref="PerIndexNormalisingPreprocessor"/> that automatically adjusts the normalisation for each index.
	/// </summary>
	[Serializable]
	public class AdaptivePerIndexNormalisingPreprocessor : BaseAdaptivePreprocessor<PerIndexNormalisingPreprocessor>
	{
		/// <inheritdoc />
		public override bool AffectsDataShape => false;

		/// <summary>
		/// Create a base adaptive preprocessor with a certain underlying preprocessor (that will be adapted).
		/// </summary>
		/// <param name="minOutputValue">The minimum output value.</param>
		/// <param name="maxOutputValue">The maximum output value.</param>
		/// <param name="sectionNames">The section names to process in this preprocessor (all if null or empty).</param>
		public AdaptivePerIndexNormalisingPreprocessor(double minOutputValue, double maxOutputValue, params string[] sectionNames)
			: base(new PerIndexNormalisingPreprocessor(minOutputValue, maxOutputValue, sectionNames, new Dictionary<int, double[]>()), sectionNames)
		{
		}

		/// <summary>
		/// Adapt the underlying preprocessor to the given array using a certain computation handler.
		/// </summary>
		/// <param name="preprocessor">The underlying preprocessor to adapt to the array.</param>
		/// <param name="array">The array.</param>
		/// <param name="handler">The computation handler.</param>
		protected override void AdaptUnderlyingPreprocessor(PerIndexNormalisingPreprocessor preprocessor, INDArray array, IComputationHandler handler)
		{
			IDictionary<int, double[]> indexMappings = preprocessor.PerIndexMinMaxMappings;
			indexMappings.Clear();

			INDArray flattenedArray = handler.FlattenTimeAndFeatures(array);

			for (int i = 0; i < array.Shape[2]; i++)
			{
				INDArray slice = handler.GetSlice(flattenedArray, 0, i, (int) flattenedArray.Shape[0], 1);

				double min = handler.Min(slice).GetValueAs<double>();
				double max = handler.Max(slice).GetValueAs<double>();

				indexMappings.Add(i, new[] { min, max, max - min });
			}
		}
	}
}
