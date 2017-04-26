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
	/// Record-wise datasets forward partial blocks, i.e. some parts of each block instead of some entire blocks per slice (as in <see cref="DatasetBlockwiseSlice"/>.
	/// </summary>
	[Serializable]
	public class DatasetRecordwiseSlice : IDataset
	{
		public string Name => UnderlyingDataset.Name;
		public bool Online
		{
			get { return UnderlyingDataset.Online; }
			set { UnderlyingDataset.Online = value; }
		}

		public IDataset UnderlyingDataset { get; }
		public double ShareOffset { get; private set; }
		public double Share { get; private set; }

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
		public int ActiveIndividualBlockRegionCount => UnderlyingDataset.ActiveIndividualBlockRegionCount;
		public int ActiveBlockRegionCount => UnderlyingDataset.ActiveBlockRegionCount;

		/// <summary>
		/// Create a record-wise slice dataset of an underlying dataset with a certain share.
		/// A record-wise split example:
		///		If the last 20% of each record should be forwarded in this dataset, share offset should be 0.8 and the share 0.2.
		///		For the first 20%, the share offset should be 0.0 and the share 0.2.
		/// </summary>
		/// <param name="underlyingDataset">The underlying dataset to slice.</param>
		/// <param name="shareOffset">The share offset.</param>
		/// <param name="share">The share.</param>
		public DatasetRecordwiseSlice(IDataset underlyingDataset, double shareOffset, double share)
		{
			if (underlyingDataset == null)
			{
				throw new ArgumentNullException(nameof(underlyingDataset));
			}

			if (shareOffset < 0.0)
			{
				throw new ArgumentException($"Share offset must be >= 0.0, but was {shareOffset}.");
			}

			if (share < 0.0)
			{
				throw new ArgumentException($"Share must be >= 0.0, but was {share}.");
			}

			if (shareOffset + share > 1.0)
			{
				throw new ArgumentException($"Share offset + share must be <= 1.0, but share offset + share was {shareOffset + share} (shareOffset: {shareOffset}, share: {share}).");
			}

			UnderlyingDataset = underlyingDataset;
			Share = share;
			ShareOffset = shareOffset;
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
			return UnderlyingDataset.IsBlockActive(blockIndex);
		}

		public bool IsBlockActive(int blockIndex, IComputationHandler handler)
		{
			return UnderlyingDataset.IsBlockActive(blockIndex, handler);
		}

		public long GetBlockSizeBytes(int blockIndex, IComputationHandler handler)
		{
			return UnderlyingDataset.GetBlockSizeBytes(blockIndex, handler);
		}

		public bool CanFetchBlocksAfter(int blockIndex)
		{
			return UnderlyingDataset.CanFetchBlocksAfter(blockIndex);
		}

		public IDictionary<string, INDArray> FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			var block = UnderlyingDataset.FetchBlock(blockIndex, handler, shouldWaitUntilAvailable);

			return block != null ? GetOwnSlice(block) : null;
		}

		protected Dictionary<string, INDArray> GetOwnSlice(IDictionary<string, INDArray> block)
		{
			Dictionary<string, INDArray> slicedBlock = new Dictionary<string, INDArray>();

			foreach (string section in block.Keys)
			{
				INDArray sectionBlock = block[section];
				long[] beginIndex = new long[sectionBlock.Rank];
				long[] endIndex = new long[sectionBlock.Rank];

				long beginRecordIndex = (long) (sectionBlock.Shape[0] * ShareOffset);
				long endRecordIndex = (long) Math.Ceiling(sectionBlock.Shape[0] * (ShareOffset + Share));

				//slice is empty, return null to indicate as specified
				if (beginRecordIndex == endRecordIndex)
				{
					return null;
				}

				beginIndex[0] = beginRecordIndex;
				endIndex[0] = endRecordIndex;

				for (int i = 1; i < sectionBlock.Rank; i++)
				{
					endIndex[i] = sectionBlock.Shape[i];
				}

				slicedBlock.Add(section, sectionBlock.Slice(beginIndex, endIndex));
			}

			return slicedBlock;
		}

		public async Task<IDictionary<string, INDArray>> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			return await Task.Run(() => FetchBlock(blockIndex, handler, shouldWaitUntilAvailable));
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			UnderlyingDataset.FreeBlock(blockIndex, handler);
		}

		public void Dispose()
		{
			UnderlyingDataset.Dispose();
		}
	}
}
