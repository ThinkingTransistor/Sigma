/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Datasets
{
	/// <summary>
	/// A dataset slice representing a part of an underlying dataset. 
	/// Block-wise datasets forward some blocks per block, i.e. some entire blocks per slice instead of some parts of each block (as in <see cref="DatasetRecordwiseSlice"/>.
	/// </summary>
	[Serializable]
	public class DatasetBlockwiseSlice : IDataset
	{
		public string Name => UnderlyingDataset.Name;
		public bool Online
		{
			get { return UnderlyingDataset.Online; }
			set { UnderlyingDataset.Online = value; }
		}

		public IDataset UnderlyingDataset { get; }
		public int SplitBeginIndex { get; }
		public int SplitEndIndex { get; }
		public int SplitSize { get; }
		public int SplitInterval { get; }

		public int TargetBlockSizeRecords => UnderlyingDataset.TargetBlockSizeRecords;
		public int MaxConcurrentActiveBlocks => UnderlyingDataset.MaxConcurrentActiveBlocks;
		public long MaxTotalActiveBlockSizeBytes => UnderlyingDataset.MaxTotalActiveBlockSizeBytes;
		public long TotalActiveBlockSizeBytes => UnderlyingDataset.TotalActiveBlockSizeBytes;
		public int MaxBlocksInCache
		{
			get { return UnderlyingDataset.MaxBlocksInCache; }
			set { UnderlyingDataset.MaxBlocksInCache = value; }
		}
		public long MaxBytesInCache
		{
			get { return UnderlyingDataset.MaxBytesInCache; }
			set { UnderlyingDataset.MaxBytesInCache = value; }
		}
		public string[] SectionNames => UnderlyingDataset.SectionNames;
		public IReadOnlyCollection<int> ActiveBlockIndices => UnderlyingDataset.ActiveBlockIndices;
		public int ActiveIndividualBlockCount => UnderlyingDataset.ActiveIndividualBlockCount;
		public int ActiveBlockRegionCount => UnderlyingDataset.ActiveBlockRegionCount;

		/// <summary>
		/// Create a block-wise slice dataset of an underlying dataset with a certain split.
		/// A block-wise split example:
		///		In order to assign a slice of 1 out of 5 blocks to this dataset, the split begin index would be 0, split end index 1 and split interval 5.
		///		Or split begin of 1 and end of 2, begin of 2 and end of 3, begin of 3 and end of 4, depending on the order in which this slice should take blocks.
		/// </summary>
		/// <param name="underlyingDataset">The underlying dataset to slice.</param>
		/// <param name="splitBeginIndex">The begin split index within the interval (inclusive).</param>
		/// <param name="splitEndIndex">The end split index within the interval (inclusive).</param>
		/// <param name="splitInterval">The split interval.</param>
		public DatasetBlockwiseSlice(IDataset underlyingDataset, int splitBeginIndex, int splitEndIndex, int splitInterval)
		{
			if (underlyingDataset == null)
			{
				throw new ArgumentNullException(nameof(underlyingDataset));
			}

			if (splitBeginIndex < 0)
			{
				throw new ArgumentException($"Split begin index must be >= 0, but was {splitBeginIndex}.");
			}

			if (splitEndIndex < 0)
			{
				throw new ArgumentException($"Split end index must be >= 0, but was {splitEndIndex}.");
			}

			if (splitBeginIndex > splitEndIndex)
			{
				throw new ArgumentException($"Split begin index must be smaller than split end index, but split begin index was {splitBeginIndex} and split end index was {splitEndIndex}.");
			}

			if (splitInterval < splitEndIndex - splitBeginIndex + 1)
			{
				throw new ArgumentException($"Split interval must be >= split size (split end index - split begin index), but split interval was {splitInterval}, split begin index {splitBeginIndex}, split end index {splitEndIndex} and split size {splitEndIndex - splitBeginIndex + 1}.");
			}

			UnderlyingDataset = underlyingDataset;
			SplitSize = splitEndIndex - splitBeginIndex + 1;
			SplitBeginIndex = splitBeginIndex;
			SplitEndIndex = splitEndIndex;
			SplitInterval = splitInterval;
		}

		/// <summary>
		/// Map an index relative to this dataset slice to an index relative to the underlying dataset.
		/// </summary>
		/// <param name="blockIndex"></param>
		/// <returns></returns>
		protected int MapToUnderlyingIndex(int blockIndex)
		{
			int round = blockIndex / SplitSize;
			int innerRoundIndex = blockIndex % SplitSize;

			return round * SplitInterval + SplitBeginIndex + innerRoundIndex;
		}

		public IDataset[] SplitBlockwise(params int[] parts)
		{
			return ExtractedDataset.SplitBlockwise(this, parts);
		}

		public IDataset[] SplitRecordwise(params double[] parts)
		{
			return ExtractedDataset.SplitRecordwise(this, parts);
		}

		public bool TrySetBlockSize(int blockSizeRecords)
		{
			return UnderlyingDataset.TrySetBlockSize(blockSizeRecords);
		}

		public bool IsBlockActive(int blockIndex)
		{
			return UnderlyingDataset.IsBlockActive(MapToUnderlyingIndex(blockIndex));
		}

		public bool IsBlockActive(int blockIndex, IComputationHandler handler)
		{
			return UnderlyingDataset.IsBlockActive(MapToUnderlyingIndex(blockIndex), handler);
		}

		public long GetBlockSizeBytes(int blockIndex, IComputationHandler handler)
		{
			return UnderlyingDataset.GetBlockSizeBytes(MapToUnderlyingIndex(blockIndex), handler);
		}

		public bool CanFetchBlocksAfter(int blockIndex)
		{
			return UnderlyingDataset.CanFetchBlocksAfter(MapToUnderlyingIndex(blockIndex));
		}

		public IDictionary<string, INDArray> FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			return UnderlyingDataset.FetchBlock(MapToUnderlyingIndex(blockIndex), handler, shouldWaitUntilAvailable);
		}

		public Task<IDictionary<string, INDArray>> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			return UnderlyingDataset.FetchBlockAsync(MapToUnderlyingIndex(blockIndex), handler, shouldWaitUntilAvailable);
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			UnderlyingDataset.FreeBlock(MapToUnderlyingIndex(blockIndex), handler);
		}

		public void Dispose()
		{
			UnderlyingDataset.Dispose();
		}
	}
}
