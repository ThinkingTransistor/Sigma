/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System.Collections.Generic;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// An undivided data iterator which yields blocks from datasets as a whole.
	/// </summary>
	public class UndividedIterator : BaseIterator
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Create an undivided iterator for a certain dataset.
		/// </summary>
		/// <param name="dataset">The underlying dataset.</param>
		public UndividedIterator(IDataset dataset) : base(dataset)
		{
		}

		/// <summary>
		/// Create a shallow copy of this data iterator (copy all members, keep dataset).
		/// Typically used to provide workers with independent sets of data iterators for the same underlying data.
		/// </summary>
		/// <returns>A shallow copy of this data iterator.</returns>
		public override IDataIterator ShallowCopy()
		{
			return new UndividedIterator(dataset: UnderlyingDataset);
		}

		public override IEnumerable<IDictionary<string, INDArray>> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			CheckNotNull(handler, environment);

			int currentIndex = 0;

			while (true)
			{
				RequireBlocks(handler, currentIndex);

				if (_fetchedBlocks[currentIndex] == null)
				{
					break;
				}

				PrepareBlocksAsync(handler, currentIndex + 1);

				var currentBlock = _fetchedBlocks[currentIndex];

				_logger.Debug($"Yielding undivided block at index {currentIndex}.");

				yield return currentBlock;

				UnderlyingDataset.FreeBlock(currentIndex, handler);
				currentIndex++;
			}
		}
	}
}
