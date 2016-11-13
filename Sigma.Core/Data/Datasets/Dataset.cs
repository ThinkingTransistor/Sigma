/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Handlers;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Datasets
{
	/// <summary>
	/// A default implementation of the IDataset interface. 
	/// Provides caching of entire blocks and reader data, partial extraction, unordered extraction, automatic block sizing, smart block loading. 
	/// </summary>
	public class Dataset : IDataset
	{
		private ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Automatically size blocks according to estimated data metrics (e.g. physical memory available, record size).
		/// </summary>
		public const int BLOCK_SIZE_AUTO = -1;

		/// <summary>
		/// Assign all available data to the first block (one block fits it all - literally).
		/// </summary>
		public const int BLOCK_SIZE_ALL = -2;

		public int MaxConcurrentActiveBlocks { get; private set; } = 24; //24 seems like a good number, right?

		public long MaxTotalActiveBlockSizeBytes { get; private set; } = SystemInformationUtils.GetAvailablePhysicalMemoryBytes() / 2; //default to half the available physical memory

		public IReadOnlyCollection<int> ActiveBlockIndices { get { return activeBlocks.Keys.ToList<int>(); } }

		public string Name { get; private set; }

		public int ActiveBlockRegionCount { get { return activeBlocks.Count; } }

		public int ActiveIndividualBlockCount { get { return activeBlocks.Values.Sum(set => set.Count); } }

		public int TargetBlockSizeRecords { get; private set; }

		public string[] SectionNames { get; private set; }

		public long TotalActiveBlockSizeBytes { get; private set; }

		public long TotalActiveRecords { get; private set; }

		public int MaxBlocksInCache { get; set; } = int.MaxValue;

		public long MaxBytesInCache { get; set; } = long.MaxValue;

		/// <summary>
		/// Indicate whether this dataset should cache the raw reader data. 
		/// If disabled, only extracted data will be cached and once processed, it might be impossible to retrieve preceding record blocks (reader streams are assumed to be non-seekable).
		/// </summary>
		public bool AllowRawReadDataCaching { get; set; } = true;

		private Dictionary<int, ISet<RecordBlock>> activeBlocks;
		private Dictionary<int, ISet<RecordBlock>> cachedBlocks;
		private ICacheProvider cacheProvider;

		private int lastReadRawDataBlockIndex = -1;
		private long totalCachedBlockSizeBytes;
		private int lastAvailableBlockIndex = Int32.MaxValue;
		private ISet<IRecordExtractor> recordExtractors;

		private Semaphore availableBlocksSemaphore;

		/// <summary>
		/// Create a dataset with a certain unique name and the record extractors to use. 
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="blockSizeRecords">The target block size for records. May also be <see cref="BLOCK_SIZE_AUTO"/> or <see cref="BLOCK_SIZE_ALL"/>.</param>
		public Dataset(string name, params IRecordExtractor[] recordExtractors) : this(name, BLOCK_SIZE_AUTO, recordExtractors)
		{
		}

		/// <summary>
		/// Create a dataset with a certain unique name, target block size in records and the record extractors to use.
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="blockSizeRecords">The target block size for records. May also be <see cref="BLOCK_SIZE_AUTO"/> or <see cref="BLOCK_SIZE_ALL"/>.</param>
		/// <param name="recordExtractors">The record extractors to fetch the data from, which provide the dataset with ready to use record blocks.</param>
		public Dataset(string name, int blockSizeRecords, params IRecordExtractor[] recordExtractors)
			: this(name, blockSizeRecords, new DiskCacheProvider(SigmaEnvironment.Globals.Get<string>("cache") + name), true, recordExtractors)
		{
		}

		/// <summary>
		/// Create a dataset with a certain unique name, target block size in records, specific cache provider and the record extractors to use.
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="blockSizeRecords">The target block size for records. May also be <see cref="BLOCK_SIZE_AUTO"/> or <see cref="BLOCK_SIZE_ALL"/>.</param>
		/// <param name="cacheProvider">The cache provider to use for caching record blocks and raw reader data.</param>
		/// <param name="flushCache">Indicate whether the cache provider should be flushed (cleared) before use. Only disable if block size and extractors used do not change (otherwise undefined behaviour).</param>
		/// <param name="recordExtractors">The record extractors to fetch the data from, which provide the dataset with ready to use record blocks.</param>
		public Dataset(string name, int blockSizeRecords, ICacheProvider cacheProvider, bool flushCache = true, params IRecordExtractor[] recordExtractors)
		{
			if (name == null)
			{
				throw new ArgumentNullException("Name cannot be null.");
			}

			if (recordExtractors == null)
			{
				throw new ArgumentNullException("Record extractors cannot be null.");
			}

			if (recordExtractors.Length == 0)
			{
				throw new ArgumentException("Datasets require at least one record extractor, but none were given.");
			}

			if (cacheProvider == null)
			{
				throw new ArgumentNullException("Cache provider cannot be null.");
			}

			if (blockSizeRecords == BLOCK_SIZE_ALL)
			{
				//just set to maximum amount of records, extracting returns the maximum available anyway and we can't know the actual availability yet
				this.TargetBlockSizeRecords = Int32.MaxValue;
			}
			else if (blockSizeRecords == BLOCK_SIZE_AUTO)
			{
				//somewhat temporary guesstimate, should probably expose the individual parameters 
				long estimatedRecordSizeBytes = 1024;
				double memoryToConsume = 0.5f;
				long optimalNumberBlocks = 24;
				long availableSystemMemory = SystemInformationUtils.GetAvailablePhysicalMemoryBytes();

				this.TargetBlockSizeRecords = (int) (availableSystemMemory * memoryToConsume / estimatedRecordSizeBytes / optimalNumberBlocks);
			}
			else if (blockSizeRecords == 0 || blockSizeRecords < -2)
			{
				throw new ArgumentException($"Block size in records must be either BLOCK_SIZE_ALL, BLOCK_SIZE_AUTO or > 0, but given block size was {blockSizeRecords}.");
			}
			else
			{
				this.TargetBlockSizeRecords = blockSizeRecords;
			}

			this.Name = name;
			this.AnalyseExtractors(recordExtractors);

			this.cacheProvider = cacheProvider;
			this.recordExtractors = new HashSet<IRecordExtractor>(recordExtractors);

			this.availableBlocksSemaphore = new Semaphore(MaxConcurrentActiveBlocks, MaxConcurrentActiveBlocks);

			this.activeBlocks = new Dictionary<int, ISet<RecordBlock>>();
			this.cachedBlocks = new Dictionary<int, ISet<RecordBlock>>();

			if (flushCache)
			{
				logger.Info($"Flushing all caches for dataset {Name} as flushCache flag was set...");

				InvalidateAndClearCaches();

				logger.Info($"Done flushing all caches for dataset {Name}.");
			}
		}

		private void AnalyseExtractors(IRecordExtractor[] extractors)
		{
			ISet<string> sectionNames = new HashSet<string>();

			int index = 0;
			foreach (IRecordExtractor extractor in extractors)
			{
				if (extractor == null)
				{
					throw new ArgumentNullException($"Extractor at index {index} was null.");
				}

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

				index++;
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
				if (block.Count == 0)
				{
					throw new InvalidOperationException($"Fetched block did not contain any named elements (was empty; is the extractor output correct?).");
				}

				availableBlocksSemaphore.WaitOne();

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

			if (IsBlockActive(blockIndex, handler))
			{
				//block already registered as active, nothing to do here
				return;
			}

			RecordBlock recordBlock = new RecordBlock(block, blockIndex, firstNamedBlock.Shape[0], handler.GetSizeBytes(block.Values.ToArray()), handler);

			recordBlock.loadedAndActive = true;

			lock (this)
			{
				TotalActiveBlockSizeBytes += recordBlock.estimatedSizeBytes;
				TotalActiveRecords += recordBlock.numberRecords;

				if (!activeBlocks.ContainsKey(blockIndex))
				{
					activeBlocks.Add(blockIndex, new HashSet<RecordBlock>());
				}

				activeBlocks[blockIndex].Add(recordBlock);
			}
		}

		private void DeregisterActiveBlock(RecordBlock recordBlock)
		{
			if (!IsBlockActive(recordBlock.blockIndex, recordBlock.handler))
			{
				//block that should be deregistered is not even registered
				return;
			}

			lock (this)
			{
				TotalActiveBlockSizeBytes -= recordBlock.estimatedSizeBytes;
				TotalActiveRecords -= recordBlock.numberRecords;

				activeBlocks[recordBlock.blockIndex].Remove(recordBlock);

				if (activeBlocks[recordBlock.blockIndex].Count == 0)
				{
					activeBlocks.Remove(recordBlock.blockIndex);
				}
			}
		}

		private void RegisterCachedBlock(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler)
		{
			if (IsBlockCached(blockIndex, handler))
			{
				//block's already cached, nothing to do here
				return;
			}

			if (!cachedBlocks.ContainsKey(blockIndex))
			{
				cachedBlocks.Add(blockIndex, new HashSet<RecordBlock>());
			}

			RecordBlock recordBlock = new RecordBlock(null, blockIndex, block.First().Value.Shape[0], handler.GetSizeBytes(block.Values.ToArray()), handler);

			recordBlock.loadedAndActive = false;

			cachedBlocks[blockIndex].Add(recordBlock);
		}

		/// <summary>
		/// Invalidate and clear all caches associated with this dataset. 
		/// WARNING: Removing cache entries may cause certain datasets to load much more slowly or incorrectly. 
		///			 Use cases include removing cache entries for old datasets or datasets with different extractors. 
		/// </summary>
		public void InvalidateAndClearCaches()
		{
			logger.Info("Invalidating and clearning all caches...");

			this.cacheProvider.RemoveAll();

			this.cachedBlocks.Clear();

			logger.Info("Done invalidating and clearning all caches.");
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

			string blockIdentifierInCache = $"extracted.{blockIndex}.{handler.DataType.Identifier}";

			//it's already stored in the cache
			lock (cacheProvider)
			{
				if (cacheProvider.IsCached(blockIdentifierInCache))
				{
					Dictionary<string, INDArray> block = cacheProvider.Load<Dictionary<string, INDArray>>(blockIdentifierInCache);

					//if its != null we could read it correctly in the right format
					if (block != null)
					{
						//register this cache entry as a properly loaded block
						RegisterCachedBlock(block, blockIndex, handler);

						return block;
					}
				}
			}

			return LoadAndExtractRaw(blockIndex, handler);
		}

		private Dictionary<string, INDArray> LoadAndExtractRaw(int blockIndex, IComputationHandler handler)
		{
			// this cannot run concurrently as cache entries can only be read and written once without wasting resources and risking corruption of data
			lock (this)
			{
				if (blockIndex >= lastReadRawDataBlockIndex)
				{
					object[] lastRawData = null;

					for (int tempBlockIndex = lastReadRawDataBlockIndex + 1; tempBlockIndex <= blockIndex; tempBlockIndex++)
					{
						lastRawData = LoadDirect(tempBlockIndex, handler);

						//looks like we couldn't read any more blocks, maybe reached the end of the underlying source streams
						if (lastRawData == null)
						{
							return null;
						}

						if (AllowRawReadDataCaching)
						{
							cacheProvider.Store($"raw.{tempBlockIndex}", lastRawData);
						}
					}

					return ExtractDirectFrom(lastRawData, blockIndex, handler);
				}
				else
				{
					if (AllowRawReadDataCaching)
					{
						string cacheIdentifier = $"raw.{blockIndex}";

						if (!cacheProvider.IsCached(cacheIdentifier))
						{
							throw new InvalidOperationException($"Unable to load cached entry for block {blockIndex} for handler {handler}, cache entry does not exist in provider {cacheProvider}.");
						}

						return ExtractDirectFrom(cacheProvider.Load<object[]>(cacheIdentifier), blockIndex, handler);
					}
					else
					{
						throw new InvalidOperationException($"Cannot load and extract raw block with index {blockIndex} because AllowRawReadDataCaching is set to false and last read position is at {lastReadRawDataBlockIndex}.");
					}
				}
			}
		}

		private object[] LoadDirect(int blockIndex, IComputationHandler handler)
		{
			IList<object> rawDataPerExtractor = new List<object>();

			PrepareExtractors();

			foreach (IRecordExtractor extractor in recordExtractors)
			{
				object data;

				lock (extractor.Reader)
				{
					data = extractor.Reader.Read(TargetBlockSizeRecords);
				}

				//check if block reader could read anything, if not, return null
				if (data == null)
				{
					lastAvailableBlockIndex = blockIndex - 1;

					logger.Info($"Cannot load block {blockIndex} for handler {handler}, the underlying stream for extractor {extractor} is unable to retrieve any more records. End of stream most likely reached.");

					return null;
				}

				rawDataPerExtractor.Add(data);
			}

			if (blockIndex > lastReadRawDataBlockIndex)
			{
				lastReadRawDataBlockIndex = blockIndex;
			}

			return rawDataPerExtractor.ToArray();
		}

		private Dictionary<string, INDArray> ExtractDirectFrom(object[] data, int blockIndex, IComputationHandler handler)
		{
			Dictionary<string, INDArray> namedBlocks = new Dictionary<string, INDArray>();

			PrepareExtractors();

			int extractorIndex = 0;
			foreach (IRecordExtractor extractor in recordExtractors)
			{
				Dictionary<string, INDArray> subNamedBlock = extractor.ExtractHierarchicalFrom(data[extractorIndex++], TargetBlockSizeRecords, handler);

				//check if block size is 0, indicating we reached the end of the stream 
				if (subNamedBlock == null)
				{
					lastAvailableBlockIndex = blockIndex - 1;

					logger.Info($"Cannot  extract block {blockIndex} for handler {handler}, the underlying stream for extractor {extractor} is unable to retrieve any more records. End of stream most likely reached.");

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

			return namedBlocks;
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			if (!activeBlocks.ContainsKey(blockIndex))
			{
				logger.Info($"Unable to free block with index {blockIndex} for handler {handler} because no block with that information is currently active.");

				return;
			}

			foreach (RecordBlock block in activeBlocks[blockIndex])
			{
				if (object.ReferenceEquals(block.handler, handler))
				{
					logger.Info($"Freeing block with index {blockIndex} for handler {handler}...");

					CacheBlockConstrained(block.namedBlocks, blockIndex, handler);

					DeregisterActiveBlock(block);

					availableBlocksSemaphore.Release();

					logger.Info($"Done freeing block with index {blockIndex} for handler {handler}.");

					return;
				}
			}

			logger.Info($"Unable to free block with index {blockIndex} for handler {handler} because no block with that information is currently active.");
		}

		private void CacheBlockConstrained(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler)
		{
			if (cachedBlocks.ContainsKey(blockIndex))
			{
				foreach (RecordBlock cachedBlock in cachedBlocks[blockIndex])
				{
					//check if block of the same type and size is already cached, if so, return, because there is no need to cache again
					if (cachedBlock.blockIndex == blockIndex && cachedBlock.handler.IsInterchangeable(handler) && block.First().Value.Shape[0] == cachedBlock.numberRecords)
					{
						logger.Info($"Skipping cache request of block {blockIndex} for handler {handler} because interchangeable block of same index, format and size is already cached.");

						return;
					}
				}
			}

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

			string cacheIdentifier = $"extracted.{blockIndex}.{handler.DataType.Identifier}";

			cacheProvider.Store(cacheIdentifier, block);

			RegisterCachedBlock(block, blockIndex, handler);

			totalCachedBlockSizeBytes += blockSizeBytes;
		}

		private void PrepareExtractors()
		{
			foreach (IRecordExtractor extractor in recordExtractors)
			{
				lock (extractor)
				{
					extractor.Prepare();
				}
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

		private bool IsBlockCached(int blockIndex, IComputationHandler handler)
		{
			if (!cachedBlocks.ContainsKey(blockIndex))
			{
				return false;
			}

			foreach (RecordBlock block in cachedBlocks[blockIndex])
			{
				if (object.ReferenceEquals(block.handler, handler))
				{
					return true;
				}
			}

			return false;
		}

		public void Dispose()
		{
			foreach (IRecordExtractor extractor in recordExtractors)
			{
				extractor.Dispose();
				extractor.Reader?.Dispose();
			}

			this.cacheProvider.Dispose();
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

				//record blocks internal block can be null
				if (namedBlocks != null)
				{
					this.firstNamedBlock = namedBlocks[namedBlocks.First().Key];
				}
			}
		}
	}
}
