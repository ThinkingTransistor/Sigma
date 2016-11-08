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
using Sigma.Core.Utils;

namespace Sigma.Core.Data.Datasets
{
	/// <summary>
	/// A default implementation of the IDataset interface. 
	/// Provides caching of entire blocks and reader data, partial extraction, unordered extraction, automatic block sizing, smart block loading. 
	/// </summary>
	public class Dataset : IDataset
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		public const int BLOCK_SIZE_AUTO = -1;
		public const int BLOCK_SIZE_ALL = -2;

		public int MaxConcurrentActiveBlocks { get; private set; } = 24;

		public long MaxTotalActiveBlockSizeBytes { get; private set; } = GetAvailablePhysicalMemory() / 2; //default to half the available physical memory


		public IReadOnlyCollection<int> ActiveBlockIndices
		{
			get
			{
				return activeBlocks.Keys.ToList<int>();
			}
		}

		public string Name { get; private set; }

		public int ActiveBlockRegionCount { get { return activeBlocks.Count; } }

		public int ActiveIndividualBlockCount { get { return activeBlocks.Values.Sum(set => set.Count); } }

		public int TargetBlockSizeRecords { get; private set; }

		public string[] SectionNames { get; private set; }

		public long TotalActiveBlockSizeBytes { get; private set; }

		public long TotalActiveRecords { get; private set; }

		public int MaxBlocksInCache { get; set; }

		public long MaxBytesInCache { get; set; }

		public bool AllowRawReadDataCaching { get; set; } = true;

		private Dictionary<int, ISet<RecordBlock>> activeBlocks;
		private Dictionary<int, ISet<RecordBlock>> cachedBlocks;
		private ICacheProvider cacheProvider;

		private int lastReadRawDataBlockIndex = -1;
		private long totalCachedBlockSizeBytes;
		private int lastAvailableBlockIndex = Int32.MaxValue;
		private ISet<IRecordExtractor> recordExtractors;

		private Semaphore availableBlocksSemaphore;
		private Semaphore takenBlocksSemaphore;

		public Dataset(string name, params IRecordExtractor[] recordExtractors) : this(name, BLOCK_SIZE_AUTO, recordExtractors)
		{
		}

		public Dataset(string name, int blockSizeRecords, params IRecordExtractor[] recordExtractors)
			: this(name, BLOCK_SIZE_AUTO, new DiskCacheProvider(SigmaEnvironment.Globals.Get<string>("cache") + name), recordExtractors)
		{
		}

		public Dataset(string name, int blockSizeInRecords, ICacheProvider cacheProvider, params IRecordExtractor[] recordExtractors)
		{
			if (name == null)
			{
				throw new ArgumentNullException("Name cannot be null.");
			}

			if (recordExtractors.Length == 0)
			{
				throw new ArgumentException("Datasets require at least one record extractor, but none were given.");
			}

			if (cacheProvider == null)
			{
				throw new ArgumentNullException("Cache provider cannot be null.");
			}

			if (blockSizeInRecords == BLOCK_SIZE_ALL)
			{
				//just set to maximum amount of records, extracting returns the maximum available anyway and we can't know the actual availability yet
				this.TargetBlockSizeRecords = Int32.MaxValue;
			}
			else if (blockSizeInRecords == BLOCK_SIZE_AUTO)
			{
				//somewhat temporary guesstimate, should probably expose the individual parameters 
				long estimatedRecordSizeBytes = 1024;
				double memoryToConsume = 0.5f;
				long optimalNumberBlocks = 24;
				long availableSystemMemory = GetAvailablePhysicalMemory();

				this.TargetBlockSizeRecords = (int) (availableSystemMemory * memoryToConsume / estimatedRecordSizeBytes / optimalNumberBlocks);
			}
			else if (blockSizeInRecords == 0 || blockSizeInRecords < -2)
			{
				throw new ArgumentException($"Block size in records must be either BLOCK_SIZE_ALL, BLOCK_SIZE_AUTO or > 0, but given block size was {blockSizeInRecords}.");
			}
			else
			{
				this.TargetBlockSizeRecords = blockSizeInRecords;
			}

			this.AnalyseExtractors(recordExtractors);

			this.recordExtractors = new HashSet<IRecordExtractor>(recordExtractors);
			this.cacheProvider = cacheProvider;

			this.availableBlocksSemaphore = new Semaphore(MaxConcurrentActiveBlocks, MaxConcurrentActiveBlocks);
			this.takenBlocksSemaphore = new Semaphore(0, MaxConcurrentActiveBlocks);

			this.activeBlocks = new Dictionary<int, ISet<RecordBlock>>();
			this.cachedBlocks = new Dictionary<int, ISet<RecordBlock>>();
		}

		private void AnalyseExtractors(IRecordExtractor[] extractors)
		{
			ISet<string> sectionNames = new HashSet<string>();

			foreach (IRecordExtractor extractor in extractors)
			{
				if (extractor.SectionNames == null)
				{
					throw new ArgumentNullException($"Section names field in extractor {extractor} was null (field has to be set by extractor).");
				}

				string[] extractorSectionNames = extractor.SectionNames;

				foreach (string sectionName in extractorSectionNames)
				{
					if (sectionNames.Contains(sectionName))
					{
						throw new ArgumentException($"Section name collision: duplicate section name {sectionName} detected for extractor {extractor}.");
					}
					else
					{
						sectionNames.Add(sectionName);
					}
				}
			}

			this.SectionNames = sectionNames.ToArray();
		}

		public bool CanFetchBlocksAfter(int blockIndex)
		{
			return blockIndex <= lastAvailableBlockIndex;
		}

		public async Task<Dictionary<string, INDArray>> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			//TODO check if block even could be fetched to not waste thread resources if shouldWaitUntilAvailable is false anyway

			return await Task.Run<Dictionary<string, INDArray>>(() => FetchBlock(blockIndex, handler, shouldWaitUntilAvailable));
		}

		public Dictionary<string, INDArray> FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			Dictionary<string, INDArray> block = FetchBlockConstrained(blockIndex, handler);

			//block could be fetched directly without violating any constraints, return successfully
			if (block != null)
			{
				RegisterActiveBlock(block, blockIndex, handler);

				return block;
			}
			else
			{
				if (shouldWaitUntilAvailable)
				{
					logger.Info($"Could not directly load block with index {blockIndex} for handler {handler} and shouldWaitUntilAvailable flag is set to true, waiting for available space...");

					return FetchBlockWhenAvailable(blockIndex, handler);
				}
				else
				{
					return null;
				}
			}
		}

		private void RegisterActiveBlock(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler)
		{
			INDArray firstNamedBlock = block[block.First().Key];

			RecordBlock recordBlock = new RecordBlock(block, blockIndex, firstNamedBlock.Shape[0], handler.GetSizeBytes(block.Values.ToArray()), handler);

			TotalActiveBlockSizeBytes += recordBlock.estimatedSizeBytes;
			TotalActiveRecords += recordBlock.numberRecords;

			if (!activeBlocks.ContainsKey(blockIndex))
			{
				activeBlocks.Add(blockIndex, new HashSet<RecordBlock>());
			}

			activeBlocks[blockIndex].Add(recordBlock);
		}

		private void DeregisterActiveBlock(RecordBlock recordBlock)
		{
			TotalActiveBlockSizeBytes -= recordBlock.estimatedSizeBytes;
			TotalActiveRecords -= recordBlock.numberRecords;

			activeBlocks[recordBlock.blockIndex].Remove(recordBlock);

			if (activeBlocks[recordBlock.blockIndex].Count == 0)
			{
				activeBlocks.Remove(recordBlock.blockIndex);
			}
		}

		private Dictionary<string, INDArray> FetchBlockWhenAvailable(int blockIndex, IComputationHandler handler)
		{
			//vanity boolean so you don't have to look at while(true)
			bool fetchedBlockSuccessfully = false;

			while (!fetchedBlockSuccessfully)
			{
				availableBlocksSemaphore.WaitOne();

				logger.Info($"Block region for request for block index {blockIndex} for handler {handler} became available, attempting to extract block to check if it fits all constraints...");

				Dictionary<string, INDArray> block = FetchBlockConstrained(blockIndex, handler);

				//if block != null we could fetch the block successfully without violating any constraints
				if (block != null)
				{
					RegisterActiveBlock(block, blockIndex, handler);

					takenBlocksSemaphore.Release();

					return block;
				}
				else
				{
					//we cannot retrieve any more blocks and shouldn't keep trying 
					if (blockIndex >= lastAvailableBlockIndex)
					{
						return null;
					}
					else
					{
						logger.Info($"Request for block with index {blockIndex} for handler {handler} was returned to the queue, waiting for available space...");

						availableBlocksSemaphore.Release();
					}
				}
			}

			//but but visual studio, this code is by definition unreachable, don't you see?
			throw new InvalidOperationException("This should really never happen. Like really. It's by definition unreachable code.");
			//apparently not
		}

		private Dictionary<string, INDArray> FetchBlockConstrained(int blockIndex, IComputationHandler handler)
		{
			if (ActiveIndividualBlockCount >= MaxConcurrentActiveBlocks)
			{
				logger.Info($"Unable to fetch block due to MaxConcurrentActiveBlocks constraint of {MaxConcurrentActiveBlocks}.");

				return null;
			}

			Dictionary<string, INDArray> block = LoadAndExtractBlock(blockIndex, handler);

			//there was nothing to load and extract, most likely end of stream
			if (block == null)
			{
				return null;
			}

			long blockSizeBytes = handler.GetSizeBytes(block.Values.ToArray());

			if (TotalActiveBlockSizeBytes + blockSizeBytes > MaxTotalActiveBlockSizeBytes)
			{
				logger.Info($"Unable to keep requested block {blockIndex} for handler {handler} in memory due to MaxTotalActiveBlockSizeBytes constraint of {MaxTotalActiveBlockSizeBytes} bytes (block of size {blockSizeBytes} would exceed constraint by {TotalActiveBlockSizeBytes + blockSizeBytes - MaxTotalActiveBlockSizeBytes} bytes.).");

				CacheBlockConstrained(block, blockIndex, handler);

				return null;
			}

			return block;
		}

		private Dictionary<string, INDArray> LoadAndExtractBlock(int blockIndex, IComputationHandler handler)
		{
			//this method takes care of
			//	- checking whether the index is already loaded and active and then converts it
			//  - or checking whether the index is already cached in the right format and loads
			//  - or if none of that, loads and extracts from the original extractors 

			//check whether a block with the same index is already active 
			if (activeBlocks.ContainsKey(blockIndex))
			{
				RecordBlock bestMatchedBlock = null;
				foreach (RecordBlock otherBlock in activeBlocks[blockIndex])
				{
					if (handler.CanConvert(otherBlock.firstNamedBlock, otherBlock.handler))
					{
						if (handler.IsInterchangeable(otherBlock.handler))
						{
							//no need to look any further, we already found the perfect match and can return without conversion
							return otherBlock.namedBlocks;
						}

						bestMatchedBlock = otherBlock;
					}
				}

				if (bestMatchedBlock != null)
				{
					//we can just convert from another already loaded block with that index
					Dictionary<string, INDArray> convertedNamedBlocks = new Dictionary<string, INDArray>();

					foreach (string name in bestMatchedBlock.namedBlocks.Keys)
					{
						convertedNamedBlocks.Add(name, handler.Convert(bestMatchedBlock.namedBlocks[name], handler));
					}

					return convertedNamedBlocks;
				}
			}

			string blockIdentifierInCache = $"extracted.{blockIndex}.{handler.DataType}.cache";

			//it's already stored in the cache in the correct format
			if (cacheProvider.IsCached(blockIdentifierInCache))
			{
				return cacheProvider.Load<Dictionary<string, INDArray>>(blockIdentifierInCache);
			}

			return LoadAndExtractRaw(blockIndex, handler);
		}

		private Dictionary<string, INDArray> LoadAndExtractRaw(int blockIndex, IComputationHandler handler)
		{
			if (blockIndex >= lastReadRawDataBlockIndex)
			{
				for (int tempBlockIndex = lastReadRawDataBlockIndex + 1; tempBlockIndex < blockIndex; tempBlockIndex++)
				{
					Dictionary<string, INDArray> block = LoadAndExtractRawDirect(tempBlockIndex, handler);

					//looks like we couldn't read any more blocks
					if (block == null)
					{
						return null;
					}
					
					if (AllowRawReadDataCaching)
					{
						cacheProvider.Store($"raw.{tempBlockIndex}.cache", block);
					}
				}

				return LoadAndExtractRawDirect(blockIndex, handler);
			}
			else
			{
				if (AllowRawReadDataCaching)
				{
					string cacheIdentifier = $"raw.{blockIndex}.cache";

					if (!cacheProvider.IsCached(cacheIdentifier))
					{
						throw new InvalidOperationException($"Unable to load cached entry for block {blockIndex} for handler {handler}, cache entry does not exist in provider {cacheProvider}.");
					}

					return cacheProvider.Load<Dictionary<string, INDArray>>(cacheIdentifier);
				}
				else
				{
					throw new InvalidOperationException($"Cannot load and extract raw block with index {blockIndex} because AllowRawReadDataCaching is set to false and last read position is at {lastReadRawDataBlockIndex}.");
				}
			}
		}

		private Dictionary<string, INDArray> LoadAndExtractRawDirect(int blockIndex, IComputationHandler handler)
		{
			Dictionary<string, INDArray> namedBlocks = new Dictionary<string, INDArray>();

			PrepareExtractors();

			foreach (IRecordExtractor extractor in recordExtractors)
			{
				Dictionary<string, INDArray> subNamedBlock = extractor.Extract(TargetBlockSizeRecords, handler);

				//check if block size is 0, indicating we reached the end of the stream 
				if (subNamedBlock == null)
				{
					lastAvailableBlockIndex = blockIndex - 1;

					logger.Info($"Cannot load and extract block {blockIndex} for handler {handler}, the underlying stream for extractor {extractor} is unable to retrieve any more records. End of stream most likely reached.");

					return null;
				}

				foreach (string name in subNamedBlock.Keys)
				{
					if (namedBlocks.ContainsKey(name))
					{
						throw new ArgumentException($"Section name collision: {name} is already used by another extractor, current extractor {extractor} cannot use it again.");
					}
					else
					{
						namedBlocks.Add(name, subNamedBlock[name]);
					}
				}
			}

			if (blockIndex > lastReadRawDataBlockIndex)
			{
				lastReadRawDataBlockIndex = blockIndex;
			}

			return namedBlocks;
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			if (!activeBlocks.ContainsKey(blockIndex))
			{
				return;
			}

			foreach (RecordBlock block in activeBlocks[blockIndex])
			{
				if (object.ReferenceEquals(block.handler, handler))
				{
					takenBlocksSemaphore.WaitOne();

					CacheBlockConstrained(block.namedBlocks, blockIndex, handler);

					DeregisterActiveBlock(block);

					availableBlocksSemaphore.Release();

					break;
				}
			}
		}

		private void CacheBlockConstrained(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler)
		{
			long blockSizeBytes = handler.GetSizeBytes(block.Values.ToArray());

			if (cachedBlocks.Count >= MaxBlocksInCache)
			{
				logger.Info($"Unable to cache block {blockIndex} for handler {handler} due to MaxBlocksInCache constraint of {MaxBlocksInCache}.");

				return;
			}

			if (blockSizeBytes + totalCachedBlockSizeBytes >= MaxBytesInCache)
			{
				logger.Info($"Unable to cache block {blockIndex} for handler {handler} due to MaxBytesInCache constraint of {MaxBytesInCache} bytes (block of size {blockSizeBytes} would exceed constraint by {totalCachedBlockSizeBytes + blockSizeBytes - MaxBytesInCache} bytes).");

				return;
			}

			string cacheIdentifier = $"extracted.{blockIndex}.{handler.DataType}.cache";

			cacheProvider.Store(cacheIdentifier, block);



			totalCachedBlockSizeBytes += blockSizeBytes;
		}

		private void PrepareExtractors()
		{
			foreach (IRecordExtractor extractor in recordExtractors)
			{
				extractor.Prepare();
			}
		}

		public long GetBlockSizeBytes(int blockIndex, IComputationHandler handler)
		{
			if (!activeBlocks.ContainsKey(blockIndex))
			{
				return -1L;
			}

			foreach (RecordBlock block in activeBlocks[blockIndex])
			{
				if (object.ReferenceEquals(block.handler, handler))
				{
					return block.estimatedSizeBytes;
				}
			}

			return -1L;
		}

		public bool IsBlockActive(int blockIndex)
		{
			return activeBlocks.ContainsKey(blockIndex);
		}

		public bool IsBlockActive(int blockIndex, IComputationHandler handler)
		{
			if (!activeBlocks.ContainsKey(blockIndex))
			{
				return false;
			}

			foreach (RecordBlock block in activeBlocks[blockIndex])
			{
				if (object.ReferenceEquals(block.handler, handler))
				{
					return true;
				}
			}

			return false;
		}

		private static long GetAvailablePhysicalMemory()
		{
			return Convert.ToInt64(new Microsoft.VisualBasic.Devices.ComputerInfo().AvailablePhysicalMemory);
		}

		internal class RecordBlock
		{
			internal Dictionary<string, INDArray> namedBlocks;
			internal INDArray firstNamedBlock;
			internal IComputationHandler handler;
			internal bool loadedAndActive;
			internal int blockIndex;
			internal long numberRecords;
			internal long estimatedSizeBytes;

			public RecordBlock(Dictionary<string, INDArray> namedBlocks, int blockIndex, long numberRecords, long estimatedSizeBytes, IComputationHandler handler)
			{
				this.namedBlocks = namedBlocks;
				this.blockIndex = blockIndex;
				this.numberRecords = numberRecords;
				this.estimatedSizeBytes = estimatedSizeBytes;
				this.handler = handler;

				this.firstNamedBlock = namedBlocks[namedBlocks.First().Key];
			}
		}
	}
}
