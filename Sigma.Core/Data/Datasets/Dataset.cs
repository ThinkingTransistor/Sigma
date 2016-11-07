/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using System.Collections.ObjectModel;

namespace Sigma.Core.Data.Datasets
{
	public class Dataset : IDataset
	{
		private IReadOnlyCollection<int> activeBlockIndices;
		public IReadOnlyCollection<int> ActiveBlockIndices
		{
			get
			{
				return activeBlockIndices;
			}
		}

		public int ActiveBlockRegionCount { get; private set; }

		public int ActiveIndividualBlockCount { get; private set; }

		public int MaxConcurrentActiveBlocks { get; private set; }

		public long MaxTotalActiveBlockSizeBytes { get; private set; }

		public long TargetBlockSizeBytes { get; private set; }

		public long TargetBlockSizeRecords { get; private set; }

		public string[] SectionNames { get; private set; }

		public long TotalActiveBlockSizeBytes { get; private set; }

		public long TotalActiveRecords { get; private set; }

		private Dictionary<int, ISet<RecordBlock>> activeBlocks;

		public bool CanFetchBlock(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public INDArray FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			throw new NotImplementedException();
		}

		public Task<INDArray> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			throw new NotImplementedException();
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public long[] GetBlockRegion(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public long GetBlockSizeBytes(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public bool IsBlockActive(int blockIndex)
		{
			throw new NotImplementedException();
		}

		public bool IsBlockActive(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		internal class RecordBlock
		{
			internal INDArray block;
			internal int blockIndex;
			internal long absoluteStartRecordIndex;
			internal long absoluteEndRecordIndex;
			internal long numberRecords;
			internal long estimatedSizeBytes;

			public RecordBlock(INDArray block, int blockIndex, long absoluteStartRecordIndex, long absoluteEndRecordIndex, long numberRecords, long estimatedSizeBytes)
			{
				this.block = block;
				this.blockIndex = blockIndex;
				this.absoluteEndRecordIndex = absoluteEndRecordIndex;
				this.absoluteStartRecordIndex = absoluteStartRecordIndex;
				this.numberRecords = numberRecords;
				this.estimatedSizeBytes = estimatedSizeBytes;
			}
		}
	}
}
