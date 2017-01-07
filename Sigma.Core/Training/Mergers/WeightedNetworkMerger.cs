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
	///     A <see cref="INetworkMerger" /> that specifies how important each network is.
	///     One can specify that the second network is twice as important as the third, but half as the first etc.
	/// </summary>
	public class WeightedNetworkMerger : BaseNetworkMerger
	{
		/// <summary>
		///     The sum of all weights.
		/// </summary>
		private double _sum;

		/// <summary>
		///     The weights for merging. The sum has to be correct.
		/// </summary>
		private double[] _weights;

		/// <summary>
		///     The weights for the merging. If updated without setting a reference, call <see cref="WeightsUpdated" />.
		/// </summary>
		public double[] Weights
		{
			get { return _weights; }
			set
			{
				_weights = value;
				WeightsUpdated();
			}
		}

		public WeightedNetworkMerger(params double[] weigths)
		{
			_weights = weigths;
			WeightsUpdated();
		}

		public void WeightsUpdated()
		{
			_sum = Weights.Sum();
		}

		protected override void CheckObjects(object[] objects)
		{
			if (objects.Length != Weights.Length)
			{
				throw new ArgumentException(nameof(Weights), $"Weights and objects do not match. You pass {objects.Length} networks, but have {Weights.Length} weights.");
			}
		}

		protected override object MergeDefault(object[] objects, IComputationHandler handler)
		{
			throw new InvalidOperationException($"Cannot merge {objects} because they are probably of type {objects[0].GetType()} (or maybe because they have different types).");
		}

		protected override double MergeDoubles(double[] doubles)
		{
			return doubles.Select((t, i) => t * Weights[i]).Sum() / _sum;
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

			INDArray sum = newHandler.Multiply(arrays[0], Weights[0]);

			for (int i = 1; i < arrays.Length; i++)
			{
				sum = newHandler.Add(sum, newHandler.Multiply(arrays[i], Weights[i]));
			}

			return newHandler.Divide(sum, _sum);
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

			INumber sum = newHandler.Multiply(numbers[0], Weights[0]);

			for (int i = 1; i < numbers.Length; i++)
			{
				sum = newHandler.Add(sum, newHandler.Multiply(numbers[i], Weights[i]));
			}

			return newHandler.Divide(sum, _sum);
		}
	}
}