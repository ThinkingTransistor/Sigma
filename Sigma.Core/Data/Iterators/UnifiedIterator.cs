/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

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
	/// An unified data iterator, which yields the entire available dataset as one block when yielded, regardless of block size.
	/// Note: Unified data iterators may be very performance intensive and drastically reduce system and training performance.
	/// </summary>
	public class UnifiedIterator : BaseIterator
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		private IDictionary<string, INDArray> _unifiedBlock;

		/// <summary>
		/// Create an unified data iterator for a certain dataset.
		/// </summary>
		/// <param name="dataset">The dataset to yield from.</param>
		public UnifiedIterator(IDataset dataset) : base(dataset)
		{
		}

		/// <summary>
		/// Create a shallow copy of this data iterator (copy relevant properties, keep dataset).
		/// Typically used to provide workers with independent sets of data iterators for the same underlying data.
		/// </summary>
		/// <returns>A shallow copy of this data iterator.</returns>
		public override IDataIterator ShallowCopy()
		{
			return new UnifiedIterator(dataset: UnderlyingDataset);
		}

		public override IEnumerable<IDictionary<string, INDArray>> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			CheckNotNull(handler, environment);

			// TODO populate registry with relevant parameters
			if (_unifiedBlock == null)
			{
				_logger.Debug($"First time yielding from iterator {this}, fetching and unifying all blocks from dataset...");

				_unifiedBlock = FetchAndMergeFromDataset(handler);
			}

			_logger.Debug($"Yielding unified block for handler {handler} consisting of {_unifiedBlock.First().Value.Shape[0]} records.");

			yield return _unifiedBlock;
		}

		private IDictionary<string, INDArray> FetchAndMergeFromDataset(IComputationHandler handler)
		{
			Dictionary<string, INDArray> unifiedBlock = new Dictionary<string, INDArray>();

			Dictionary<string, IList<INDArray>> allFetchedBlocks = new Dictionary<string, IList<INDArray>>();

			int currentBlockIndex = 0;
			while (true)
			{
				IDictionary<string, INDArray> currentBlock = UnderlyingDataset.FetchBlock(currentBlockIndex, handler);

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
