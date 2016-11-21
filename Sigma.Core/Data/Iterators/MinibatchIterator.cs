/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// A minibatch iterator which randomly traverses a dataset in minibatches of a certain size.
	/// </summary>
	public class MinibatchIterator : BaseIterator
	{
		/// <summary>
		/// Indicate that the minibatch size should be chosen automatically.
		/// </summary>
		public const int MinibatchSizeAuto = -1;

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly IList<int> _currentBatchNotTraversedIndices;
		private readonly ISet<int> _allPossibleIndices;
		private int _lastTraversedIndex = -1;
		private int _highestTraversedIndex = -1;
		private bool _traversedUntilEnd;

		/// <summary>
		/// Create a minibatch iterator with a certain minibatch size in records for a certain dataset.
		/// </summary>
		/// <param name="minibatchSizeRecords">The mini batch size in records (the amount of records that will be yielded per block).</param>
		/// <param name="dataset">The dataset to fetch blocks from.</param>
		public MinibatchIterator(int minibatchSizeRecords, IDataset dataset) : base(dataset)
		{
			if (minibatchSizeRecords == MinibatchSizeAuto)
			{
				const int optimalMinibatchSizeRecords = 16; //idk, 16 maybe

				//not sure yet if this is the best way to handle auto mini batch size
				if (dataset.TrySetBlockSize(optimalMinibatchSizeRecords))
				{
					minibatchSizeRecords = optimalMinibatchSizeRecords;

					_logger.Info($"Set block size in dataset {dataset} to optimal minibatch size of {optimalMinibatchSizeRecords}.");
				}
				else
				{
					minibatchSizeRecords = dataset.TargetBlockSizeRecords;

					_logger.Info($"Unable to set block size of dataset {dataset} to the optimal size of {optimalMinibatchSizeRecords}, minibatch size is dataset block size of {minibatchSizeRecords}.");
				}
			}
			else if (minibatchSizeRecords <= 0)
			{
				throw new ArgumentException($"Minibatch size records must be >= 1 or MinibatchSizeAuto ({MinibatchSizeAuto}).");
			}

			if (!dataset.TrySetBlockSize(minibatchSizeRecords))
			{
				throw new InvalidOperationException($"Unable to align minibatch size of {minibatchSizeRecords} with dataset block size of {dataset.TargetBlockSizeRecords}, block overloading with different size is not supported.");
			}

			_currentBatchNotTraversedIndices = new List<int>();
			_allPossibleIndices = new HashSet<int>();
		}

		public override Dictionary<string, INDArray> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			int yieldedIndex = -1;

			if (!_traversedUntilEnd)
			{
				RequireBlocks(handler, _highestTraversedIndex + 1);
				PrepareBlocksAsync(handler, _highestTraversedIndex + 2);

				yieldedIndex = ++_highestTraversedIndex;

				if (!_allPossibleIndices.Contains(yieldedIndex))
				{
					_allPossibleIndices.Add(yieldedIndex);
				}

				if (_fetchedBlocks[yieldedIndex] == null)
				{
					_fetchedBlocks.Remove(yieldedIndex);

					_traversedUntilEnd = true;

					_logger.Info($"Completed initial traversal of blocks until end, last currently available block seems to be {yieldedIndex}.");

					ResetNotTraversedIndices();
				}
			}

			if (_traversedUntilEnd)
			{
				yieldedIndex = _currentBatchNotTraversedIndices[environment.Random.Next(_currentBatchNotTraversedIndices.Count)];
				RequireBlocks(handler, yieldedIndex);

				// looks like the end of the dataset is still the end of the dataset, let's try again every now again 
				// (index after last possible is also in possible list for online datasets)
				if (_fetchedBlocks[yieldedIndex] == null)
				{
					_currentBatchNotTraversedIndices.Remove(yieldedIndex);

					ResetNotTraversedIndicesIfAllTraversed();

					// Count - 1 to exclude last index which was the test for online dataset index, but as it apparently still hasn't expanded, exclude it
					yieldedIndex = _currentBatchNotTraversedIndices[environment.Random.Next(_currentBatchNotTraversedIndices.Count - 1)]; 
					RequireBlocks(handler, yieldedIndex);
				}
			}

			if (yieldedIndex < 0)
			{
				throw new InvalidOperationException($"Unable to yield block for {handler}, suggested yielded index was {yieldedIndex} (internal error).");
			}

			if (_lastTraversedIndex >= 0)
			{
				UnderlyingDataset.FreeBlock(_lastTraversedIndex, handler);
			}

			_lastTraversedIndex = yieldedIndex;
			_currentBatchNotTraversedIndices.Remove(yieldedIndex);

			if (_traversedUntilEnd)
			{
				ResetNotTraversedIndicesIfAllTraversed();
			}

			_logger.Info($"Yielding block with index {yieldedIndex} for handler {handler} consisting of {_fetchedBlocks[yieldedIndex].First().Value.Shape[0]} records.");

			return _fetchedBlocks[yieldedIndex];
		}

		private void ResetNotTraversedIndicesIfAllTraversed()
		{
			if (_currentBatchNotTraversedIndices.Count == 0)
			{
				ResetNotTraversedIndices();
			}
		}

		private void ResetNotTraversedIndices()
		{
			_currentBatchNotTraversedIndices.Clear();

			foreach (int index in _allPossibleIndices)
			{
				_currentBatchNotTraversedIndices.Add(index);
			}

			_logger.Info($"Reset indices to traverse for next full batch, total of {_allPossibleIndices.Count} indices/minibatches.");
		}
	}
}
