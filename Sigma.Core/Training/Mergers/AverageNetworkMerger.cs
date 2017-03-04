/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Linq;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Training.Mergers
{
	/// <summary>
	///     A <see cref="INetworkMerger" /> that simple adds all values together and divides it by the amount (arithmetic
	///     mean).
	///     Theoretically it is equal to a <see cref="WeightedNetworkMerger" /> with equal weights.
	/// </summary>
	[Serializable]
	public class AverageNetworkMerger : BaseNetworkMerger
	{
		public AverageNetworkMerger(params string[] matchIdentifiers) : base(matchIdentifiers)
		{
		}

		protected override object MergeDefault(object[] objects, IComputationHandler handler)
		{
			throw new InvalidOperationException($"Cannot merge {objects} because they are probably of type {objects[0].GetType()} which is not supported (or maybe because the passed objects have different types).");
		}

		protected override double MergeDoubles(double[] doubles)
		{
			return doubles.Sum() / doubles.Length;
		}

		protected override INDArray MergeNDArrays(INDArray[] arrays, IComputationHandler handler)
		{
			IComputationHandler newHandler = null;
			if (handler == null)
			{
				foreach (INDArray ndArray in arrays)
				{
					if (ndArray.AssociatedHandler != null)
					{
						newHandler = ndArray.AssociatedHandler;
					}
				}
			}
			else
			{
				newHandler = handler;
			}

			if (newHandler == null)
			{
				throw new ArgumentNullException(nameof(handler));
			}

			INDArray sum = arrays[0];

			for (int i = 1; i < arrays.Length; i++)
			{
				sum = newHandler.Add(sum, arrays[i]);
			}

			return newHandler.Divide(sum, arrays.Length);
		}

		protected override INumber MergeNumbers(INumber[] numbers, IComputationHandler handler)
		{
			IComputationHandler newHandler = null;
			if (handler == null)
			{
				foreach (INumber number in numbers)
				{
					if (number.AssociatedHandler != null)
					{
						newHandler = number.AssociatedHandler;
					}
				}
			}
			else
			{
				newHandler = handler;
			}

			if (newHandler == null)
			{
				throw new ArgumentNullException(nameof(handler));
			}

			INumber sum = numbers[0];

			for (int i = 1; i < numbers.Length; i++)
			{
				sum = newHandler.Add(sum, numbers[i]);
			}

			return newHandler.Divide(sum, numbers.Length);
		}
	}
}