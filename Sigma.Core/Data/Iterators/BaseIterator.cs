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
using System.Threading.Tasks;
using Sigma.Core.Utils;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// A base iterator for data iterators working with datasets. Includes asynchronous background block preparation and fetching.
	/// </summary>
	public abstract class BaseIterator : IDataIterator
	{
		/// <summary>
		/// The underlying dataset of this data iterator.
		/// </summary>
		public IDataset UnderlyingDataset { get; }

		/// <summary>
		/// A registry containing relevant parameters of this data iterator.
		/// </summary>
		public IRegistry Registry { get; }

		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// The completely fetched blocks by block index.
		/// </summary>
		protected readonly Dictionary<int, IDictionary<string, INDArray>> _fetchedBlocks;
		private readonly Dictionary<int, Task<IDictionary<string, INDArray>>> _pendingFetchBlockTasks;

		/// <summary>
		/// Create a base iterator for a certain dataset.
		/// </summary>
		/// <param name="dataset">The dataset to fetch blocks from.</param>
		protected BaseIterator(IDataset dataset)
		{
			if (dataset == null)
			{
				throw new ArgumentNullException(nameof(dataset));
			}

			UnderlyingDataset = dataset;
			Registry = new Registry(tags: "iterator");

			_fetchedBlocks = new Dictionary<int, IDictionary<string, INDArray>>();
			_pendingFetchBlockTasks = new Dictionary<int, Task<IDictionary<string, INDArray>>>();
		}

		public abstract IEnumerable<IDictionary<string, INDArray>> Yield(IComputationHandler handler, SigmaEnvironment environment);
		public abstract IDataIterator ShallowCopy();

		protected void CheckNotNull(IComputationHandler handler, SigmaEnvironment environment)
		{
			if (handler == null)
			{
				throw new ArgumentNullException(nameof(handler));
			}

			if (environment == null)
			{
				throw new ArgumentNullException(nameof(environment));
			}
		}

		protected void RequireBlocks(IComputationHandler handler, params int[] indices)
		{
			foreach (int index in indices)
			{
				if (_fetchedBlocks.ContainsKey(index))
				{
					continue;
				}

				if (_pendingFetchBlockTasks.ContainsKey(index))
				{
					_logger.Info($"Waiting for already running asynchronous block fetch for index {index} to complete as it is now required...");

					_pendingFetchBlockTasks[index].Wait();

					IDictionary<string, INDArray> block = _pendingFetchBlockTasks[index].Result;

					_pendingFetchBlockTasks.Remove(index);

					_fetchedBlocks.Add(index, block);

					_logger.Info($"Done waiting for asynchronous block fetch for index {index} to complete, fetch completed.");
				}
				else
				{
					_logger.Info($"Fetching required block with index {index}...");

					_fetchedBlocks.Add(index, UnderlyingDataset.FetchBlock(index, handler));

					_logger.Info($"Done fetching required block with index {index}.");
				}
			}
		}

		protected void PrepareBlocksAsync(IComputationHandler handler, params int[] indices)
		{
			foreach (int index in indices)
			{
				if (!_pendingFetchBlockTasks.ContainsKey(index) && !_fetchedBlocks.ContainsKey(index))
				{
					_pendingFetchBlockTasks.Add(index, Task.Run(() =>
					{
						_logger.Info($"Started asynchronous background preparation of block with index {index}.");

						var block = UnderlyingDataset.FetchBlock(index, handler);

						_logger.Info($"Done with asynchronous background preparation of block with index {index}.");

						return block;
					}));
				}
			}
		}
	}
}
