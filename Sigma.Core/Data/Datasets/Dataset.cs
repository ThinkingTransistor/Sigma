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
using Sigma.Core.Data.Extractors;
using log4net;
using System.Threading;

namespace Sigma.Core.Data.Datasets
{
	public class Dataset : IDataset
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public const int BLOCK_SIZE_AUTO = -1;
		public const int BLOCK_SIZE_ALL = -2;

		public IReadOnlyCollection<int> ActiveBlockIndices
		{
			get
			{
				return activeBlocks.Keys.ToList<int>();
			}
		}

		public int ActiveBlockRegionCount { get; private set; }

		public int ActiveIndividualBlockCount { get; private set; }

		public int MaxConcurrentActiveBlocks { get; private set; } = 24;

		public long MaxTotalActiveBlockSizeBytes { get; private set; } = long.MaxValue;

		public long TargetBlockSizeRecords { get; private set; }

		public string[] SectionNames { get; private set; }

		public long TotalActiveBlockSizeBytes { get; private set; }

		public long TotalActiveRecords { get; private set; }

		public int MaxBlocksInCache { get; set; }

		public long MaxBytesInCache { get; set; }

		private Dictionary<int, ISet<RecordBlock>> activeBlocks;
		private Dictionary<int, ISet<RecordBlock>> cachedBlocks;

		private long totalCachedBlockSizeBytes;
		private int lastAvailableBlockIndex = Int32.MaxValue;
		private ISet<IRecordExtractor> recordExtractors;

		private Semaphore availableBlocksSemaphore;
		private Semaphore takenBlocksSemaphore;

		public Dataset(params IRecordExtractor[] recordExtractors) : this(BLOCK_SIZE_AUTO, recordExtractors)
		{
		}
		
		public Dataset(long blockSizeRecords, params IRecordExtractor[] recordExtractors)
		{
			if (blockSizeRecords == BLOCK_SIZE_ALL)
			{
				this.TargetBlockSizeRecords = long.MaxValue;
			}
			else if (blockSizeRecords == BLOCK_SIZE_AUTO)
			{
				long estimatedRecordSizeBytes = 1024;
				double memoryToConsume = 0.5f;
				long optimalNumberBlocks = 24;
				long availableSystemMemory = Convert.ToInt64(new Microsoft.VisualBasic.Devices.ComputerInfo().AvailablePhysicalMemory);

				this.TargetBlockSizeRecords = (long) (availableSystemMemory * memoryToConsume / estimatedRecordSizeBytes / optimalNumberBlocks);
			}
			else if (blockSizeRecords == 0 || blockSizeRecords < -2)
			{
				throw new ArgumentException($"Block size in records must be either BLOCK_SIZE_ALL, BLOCK_SIZE_AUTO or > 0, but given block size was {blockSizeRecords}.");
			}

			if (recordExtractors.Length == 0)
			{
				throw new ArgumentException("Datasets require at least one record extractor, but none were given.");
			}

			this.recordExtractors = new HashSet<IRecordExtractor>(recordExtractors);
			this.TargetBlockSizeRecords = blockSizeRecords;
			this.availableBlocksSemaphore = new Semaphore(MaxConcurrentActiveBlocks, MaxConcurrentActiveBlocks);
			this.takenBlocksSemaphore = new Semaphore(0, MaxConcurrentActiveBlocks);
		}

		public bool CanFetchBlocksAfter(int blockIndex)
		{
			return blockIndex <= lastAvailableBlockIndex;
		}

		public bool CanFetchBlockDirectly(int blockIndex, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public async Task<INDArray> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			//TODO check if block even could be fetched to not waste thread resources if shouldWaitUntilAvailable is false anyway

			INDArray block = await Task.Run<INDArray>(() => FetchBlock(blockIndex, handler, shouldWaitUntilAvailable));

			//TODO

			return block;
		}

		public INDArray FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			if (!CanFetchBlockDirectly(blockIndex, handler))
			{
				if (shouldWaitUntilAvailable)
				{
					logger.Info($"Could not directly load block with index {blockIndex} for handler {handler} and shouldWait flag is set to true, waiting for available space...");

					return FetchBlockWhenAvailable(blockIndex, handler);
				}
				else
				{
					return null;
				}
			}
			else
			{
				INDArray block = FetchBlockConstrained(blockIndex, handler);

				//couldn't fetch block without violating constraints
				if (block == null)
				{
					return null;
				}

				RegisterActiveBlock(block, blockIndex, handler);

				return block;
			}
		}

		private void RegisterActiveBlock(INDArray block, int blockIndex, IComputationHandler handler)
		{
			
		}

		private INDArray FetchBlockWhenAvailable(int blockIndex, IComputationHandler handler)
		{
			//vanity bool so you don't have to look at while(true)
			bool fetchedBlockSuccessfully = false;

			while (!fetchedBlockSuccessfully)
			{
				availableBlocksSemaphore.WaitOne();

				logger.Info($"Block region for request for block index {blockIndex} for handler {handler} became available, attempting to extract block to check if it fits all constraints...");

				INDArray block = FetchBlockConstrained(blockIndex, handler);

				//if block != null we could fetch the block successfully without violating any constraints
				if (block != null)
				{
					takenBlocksSemaphore.Release();

					RegisterActiveBlock(block, blockIndex, handler);

					return block;
				}
				else
				{
					logger.Info($"Request for block with index {blockIndex} for handler {handler} was returned to the queue, waiting for available space...");

					availableBlocksSemaphore.Release();
				}
			}

			//but but visual studio, this code is by definition unreachable, don't you see?
			throw new InvalidOperationException("This should really never happen. Like really. It's by definition unreachable code.");
			//apparently not
		}

		private INDArray FetchBlockConstrained(int blockIndex, IComputationHandler handler)
		{
			if (ActiveIndividualBlockCount >= MaxConcurrentActiveBlocks)
			{
				logger.Info($"Unable to fetch block due to MaxConcurrentActiveBlocks constraint of {MaxConcurrentActiveBlocks}.");

				return null;
			}

			INDArray block = LoadAndExtractBlock(blockIndex, handler);

			long blockSizeBytes = handler.GetSizeBytes(block);

			if (TotalActiveBlockSizeBytes + blockSizeBytes > MaxTotalActiveBlockSizeBytes)
			{
				logger.Info($"Unable to keep requested block {blockIndex} for handler {handler} in memory due to MaxTotalActiveBlockSizeBytes constraint of {MaxTotalActiveBlockSizeBytes} bytes (block of size {blockSizeBytes} would exceed constraint by {TotalActiveBlockSizeBytes + blockSizeBytes - MaxTotalActiveBlockSizeBytes} bytes.).");

				CacheBlockConstrained(block, blockIndex, handler);

				return null;
			}

			return block;
		}

		private INDArray LoadAndExtractBlock(int blockIndex, IComputationHandler handler)
		{
			//this method takes care of
			//	- checking whether the index is already loaded and active and then converts it
			//  - or checking whether the index is already cached and loads and converts it
			//  - or if none of that, loads and extracts from the original extractors 

			throw new NotImplementedException();
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			takenBlocksSemaphore.WaitOne();

			availableBlocksSemaphore.Release();
		}


		private void CacheBlockConstrained(INDArray block, int blockIndex, IComputationHandler handler)
		{
			long blockSize = handler.GetSizeBytes(block);
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
			internal bool loadedAndActive;
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
