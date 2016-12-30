/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;

namespace Sigma.Core.Training.Initialisers
{
	/// <summary>
	/// A base initialiser for simplified per value calculations.
	/// </summary>
	public abstract class BaseInitialiser : IInitialiser
	{
		public void Initialise(INDArray array, IComputationHandler handler, Random random)
		{
			if (array == null) throw new ArgumentNullException(nameof(array));
			if (handler == null) throw new ArgumentNullException(nameof(handler));
			if (random == null) throw new ArgumentNullException(nameof(random));

			long[] indices = new long[array.Rank];

			for (long i = 0; i < array.Length; i++)
			{
				indices = NDArrayUtils.GetIndices(i, array.Shape, array.Strides, indices);

				array.SetValue(GetValue(indices, array.Shape, random), indices);
			}
		}

		public void Initialise(INumber number, IComputationHandler handler, Random random)
		{
			if (number == null) throw new ArgumentNullException(nameof(number));
			if (handler == null) throw new ArgumentNullException(nameof(handler));
			if (random == null) throw new ArgumentNullException(nameof(random));

			number.Value = GetValue(new[] { 0L }, new[] { 1L, 1L, 1L }, random);
		}

		/// <summary>
		/// Get the value to set for certain indices shape and a helper randomiser. 
		/// </summary>
		/// <param name="indices">The indices.</param>
		/// <param name="shape">The shape.</param>
		/// <param name="random">The randomiser.</param>
		/// <returns>The value to set at the given indices.</returns>
		public abstract object GetValue(long[] indices, long[] shape, Random random);
	}
}
