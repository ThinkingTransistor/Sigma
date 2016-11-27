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
using Sigma.Core.Utils;

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

		/// <summary>
		/// The minibatch size used in this minibatch iterator.
		/// </summary>
		public int MinibatchSize { get; }

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private readonly IList<int> _currentBatchNotTraversedBlockIndices;
		private readonly ISet<int> _allAvailableBlockIndices;
		private readonly IList<int> _currentBlockNotTraversedSlices;
		private IDictionary<string, INDArray> _currentBlock;
		private long _currentBlockSizeRecords;
		private int _currentBlockIndex = -1;
		private int _currentHighestTraversedBlockIndex = -1;
		private int _totalHighestTraversedBlockIndex = -1;
		private bool _traversedAllBlocks;
		private bool _requireNewBlock = true;

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
				minibatchSizeRecords = optimalMinibatchSizeRecords;
			}
			else if (minibatchSizeRecords <= 0)
			{
				throw new ArgumentException($"Minibatch size records must be >= 1 or MinibatchSizeAuto ({MinibatchSizeAuto}).");
			}

			_currentBatchNotTraversedBlockIndices = new List<int>();
			_currentBlockNotTraversedSlices = new List<int>();
			_allAvailableBlockIndices = new HashSet<int>();

			MinibatchSize = minibatchSizeRecords;
		}

		public override IEnumerable<IDictionary<string, INDArray>> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			CheckNotNull(handler, environment);

			while (!_traversedAllBlocks || _currentBatchNotTraversedBlockIndices.Count > 0)
			{
				if (_requireNewBlock)
				{
					int yieldedIndex = YieldBlock(handler, environment);

					if (_currentBlockIndex >= 0)
					{
						UnderlyingDataset.FreeBlock(_currentBlockIndex, handler);
					}

					_currentBlockIndex = yieldedIndex;
					_currentBlock = _fetchedBlocks[_currentBlockIndex];
					_currentBatchNotTraversedBlockIndices.Remove(yieldedIndex);
					_currentBlockSizeRecords = _currentBlock.First().Value.Shape[0];

					if (_traversedAllBlocks)
					{
						ResetNotTraversedBlockIndices();

						yield break;
					}

					_currentBlockNotTraversedSlices.Clear();

					int numSlices = (int) Math.Ceiling((double) (_currentBlockSizeRecords - 1) / MinibatchSize);
					foreach (int sliceIndex in ArrayUtils.Range(0, numSlices))
					{
						_currentBlockNotTraversedSlices.Add(sliceIndex);
					}

					_requireNewBlock = false;
				}

				int index = _currentBlockNotTraversedSlices[environment.Random.Next(_currentBlockNotTraversedSlices.Count)];
				int beginRecordIndex = index * MinibatchSize;
				long endRecordIndex = Math.Min(_currentBlockSizeRecords, (index + 1) * MinibatchSize);

				_currentBlockNotTraversedSlices.Remove(index);

				if (_currentBlockNotTraversedSlices.Count == 0)
				{
					_requireNewBlock = true;
				}

				yield return SliceBlock(_fetchedBlocks[_currentBlockIndex], beginRecordIndex, endRecordIndex);
			}
		}

		private IDictionary<string, INDArray> SliceBlock(Dictionary<string, INDArray> block, int beginRecordIndex, long endRecordIndex)
		{
			IDictionary<string, INDArray> slice = new Dictionary<string, INDArray>();

			foreach (string section in block.Keys)
			{
				int rank = block[section].Rank;
				long[] beginIndices = new long[rank];
				long[] endIndices = new long[rank];

				beginIndices[0] = beginRecordIndex;
				endIndices[0] = endRecordIndex;

				for (int i = 1; i < rank; i++)
				{
					endIndices[i] = block[section].Shape[i];
				}

				slice.Add(section, block[section].Slice(beginIndices, endIndices));
			}

			return slice;
		}

		private int YieldBlock(IComputationHandler handler, SigmaEnvironment environment)
		{
			int yieldedIndex = -1;

			RequireBlocks(handler, _currentHighestTraversedBlockIndex + 1);
			PrepareBlocksAsync(handler, _currentHighestTraversedBlockIndex + 2);

			yieldedIndex = ++_currentHighestTraversedBlockIndex;

			if (_currentHighestTraversedBlockIndex > _totalHighestTraversedBlockIndex)
			{
				_totalHighestTraversedBlockIndex = _currentHighestTraversedBlockIndex;
			}

			if (!_allAvailableBlockIndices.Contains(yieldedIndex))
			{
				_allAvailableBlockIndices.Add(yieldedIndex);
			}

			if (_fetchedBlocks[yieldedIndex] == null)
			{
				_fetchedBlocks.Remove(yieldedIndex);

				_traversedAllBlocks = true;

				_logger.Info($"Completed traversal of blocks until end, last currently available block seems to be {yieldedIndex}.");

				ResetNotTraversedBlockIndices();

				_currentHighestTraversedBlockIndex = 0;
				yieldedIndex = 0;

				PrepareBlocksAsync(handler, 0);
			}

			return yieldedIndex;
		}

		private void ResetNotTraversedBlockIndices()
		{
			_currentBatchNotTraversedBlockIndices.Clear();

			foreach (int index in _allAvailableBlockIndices)
			{
				_currentBatchNotTraversedBlockIndices.Add(index);
			}

			_logger.Info($"Reset indices to traverse for next full batch, total of {_allAvailableBlockIndices.Count} indices/minibatches.");
		}
	}
}
