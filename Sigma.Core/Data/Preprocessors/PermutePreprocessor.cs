/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Preprocessors
{
	/// <summary>
	/// A permutation preprocessor for reshaping blocks along a permutation vector. 
	/// </summary>
	[Serializable]
	public class PermutePreprocessor : BasePreprocessor
	{
		/// <inheritdoc />
		public override bool AffectsDataShape => true;

		private int[] _rearrangedDimensions;

		/// <summary>
		/// Create a permutation preprocessor with certain rearranged dimensions (permute vector, 0-indexed). 
		/// </summary>
		/// <param name="rearrangedDimensions">The rearranged dimensions. See <see cref="INDArray.Permute"/>.</param>
		public PermutePreprocessor(params int[] rearrangedDimensions) : this((string[]) null, rearrangedDimensions)
		{
		}

		/// <summary>
		/// Create a permutation preprocessor with certain rearranged dimensions (permute vector, 0-indexed). 
		/// </summary>
		/// <param name="sectionName">The specific section this preprocessor should be applied to.</param>
		/// <param name="rearrangedDimensions">The rearranged dimensions. See <see cref="INDArray.Permute"/>.</param>
		public PermutePreprocessor(string sectionName, int[] rearrangedDimensions) : this(new[] { sectionName }, rearrangedDimensions)
		{
		}

		/// <summary>
		/// Create a permutation preprocessor with certain rearranged dimensions (permute vector, 0-indexed). 
		/// </summary>
		/// <param name="sectionNames">The specific sections this preprocessor should be applied to.</param>
		/// <param name="rearrangedDimensions">The rearranged dimensions. See <see cref="INDArray.Permute"/>.</param>
		public PermutePreprocessor(string[] sectionNames, int[] rearrangedDimensions) : base(sectionNames)
		{
			if (rearrangedDimensions == null) throw new ArgumentNullException(nameof(rearrangedDimensions));
			if (rearrangedDimensions.Length < 3) throw new ArgumentException($"Rearranged dimensions vector must be at least of length 3 (batch, time, features) but was {rearrangedDimensions.Length}.");

			_rearrangedDimensions = rearrangedDimensions;
		}

		/// <summary>
		/// Process a certain ndarray with a certain computation handler.
		/// </summary>
		/// <param name="array">The ndarray to process.</param>
		/// <param name="handler">The computation handler to do the processing with.</param>
		/// <returns>An ndarray with the processed contents of the given array (can be the same or a new one).</returns>
		internal override INDArray ProcessDirect(INDArray array, IComputationHandler handler)
		{
			array.PermuteSelf(_rearrangedDimensions);

			return array;
		}
	}
}
