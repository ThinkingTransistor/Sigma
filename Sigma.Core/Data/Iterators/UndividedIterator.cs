/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using log4net;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// An undivided data iterator, which yields the entire available dataset as one block when yielded.
	/// Note: Undivided data iterators may be very performance intensive and drastically use system and training performance.
	/// </summary>
	public class UndividedIterator : BaseIterator
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private Dictionary<string, INDArray> _unifiedBlock;

		/// <summary>
		/// Create an undivided data iterator for a certain dataset.
		/// </summary>
		/// <param name="dataset">The dataset to yield from.</param>
		public UndividedIterator(IDataset dataset) : base(dataset)
		{
		}

		public override IEnumerable<IDictionary<string, INDArray>> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			if (_unifiedBlock == null)
			{
				_logger.Info("First time yielding from this iterator, fetching and unifying all blocks from dataset...");

				_unifiedBlock = FetchAndMergeFromDataset(handler);
			}

			_logger.Info($"Yielding undivided block for handler {handler} consisting of {_unifiedBlock.First().Value.Shape[0]} records.");

			yield return _unifiedBlock;
		}

		private Dictionary<string, INDArray> FetchAndMergeFromDataset(IComputationHandler handler)
		{
			Dictionary<string, INDArray> unifiedBlock = new Dictionary<string, INDArray>();

			Dictionary<string, IList<INDArray>> allFetchedBlocks = new Dictionary<string, IList<INDArray>>();

			int currentBlockIndex = 0;
			while (true)
			{
				Dictionary<string, INDArray> currentBlock = UnderlyingDataset.FetchBlock(currentBlockIndex, handler);

				if (currentBlock != null)
				{
					foreach (string section in currentBlock.Keys)
					{
						if (!allFetchedBlocks.ContainsKey(section))
						{
							allFetchedBlocks.Add(section, new List<INDArray>());
						}

						allFetchedBlocks[section].Add(currentBlock[section]);
					}

					UnderlyingDataset.FreeBlock(currentBlockIndex, handler);
				}

				if (!UnderlyingDataset.CanFetchBlocksAfter(currentBlockIndex))
				{
					break;
				}

				currentBlockIndex++;
			}

			if (allFetchedBlocks.Count == 0)
			{
				throw new InvalidOperationException($"Cannot fetch and merge an empty block list, no blocks could be fetched from the dataset {UnderlyingDataset} for handler {handler}.");
			}

			foreach (string section in allFetchedBlocks.Keys)
			{
				unifiedBlock.Add(section, handler.MergeBatch(allFetchedBlocks[section].ToArray()));
			}

			return unifiedBlock;
		}
	}
}
