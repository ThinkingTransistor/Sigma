using System;
using System.Linq;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Mergers
{
	/// <summary>
	/// A network manager that uses the highest value. 
	/// </summary>
	public class MaxNetworkMerger : BaseNetworkMerger
	{
		/// <summary>
		///     This method is used to merge doubles.
		/// </summary>
		/// <param name="doubles">The doubles.</param>
		/// <returns>The merged value.</returns>
		protected override double MergeDoubles(double[] doubles)
		{
			return doubles.Max();
		}

		/// <summary>
		///     This method is used to merge <see cref="INDArray" />s.
		/// </summary>
		/// <param name="arrays">The arrays to merge. </param>
		/// <param name="handler">The handler that may or may not be specified.</param>
		/// <returns>A merged <see cref="INDArray" />.</returns>
		protected override INDArray MergeNDArrays(INDArray[] arrays, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		/// <summary>
		///     This method is used to merge <see cref="INDArray" />s.
		/// </summary>
		/// <param name="numbers">The numbers to merge. </param>
		/// <param name="handler">The handler that may or may not be specified.</param>
		/// <returns>A merged <see cref="INumber" />.</returns>
		protected override INumber MergeNumbers(INumber[] numbers, IComputationHandler handler)
		{
			throw new NotImplementedException();
			//? return numbers.Max();
		}
	}
}