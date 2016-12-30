/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using log4net;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Utils;

namespace Sigma.Core.Data.Datasets
{
	/// <summary>
	/// A default implementation of the IDataset interface. 
	/// Provides caching of entire blocks and reader data, partial extraction, unordered extraction, automatic block sizing, smart block loading. 
	/// </summary>
	public class Dataset : IDataset
	{
		private readonly ILog _logger = LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);

		/// <summary>
		/// Automatically size blocks according to estimated data metrics (e.g. physical memory available, record size).
		/// </summary>
		public const int BlockSizeAuto = -1;

		/// <summary>
		/// Assign all available data to the first block (one block fits it all - literally).
		/// </summary>
		public const int BlockSizeAll = -2;

		public int MaxConcurrentActiveBlocks { get; } = 24; //24 seems like a good number, right?

		public long MaxTotalActiveBlockSizeBytes { get; } = SystemInformationUtils.GetAvailablePhysicalMemoryBytes() / 2; //default to half the available physical memory

		public IReadOnlyCollection<int> ActiveBlockIndices => _activeBlocks.Keys.ToList();

		public string Name { get; }

		public int ActiveBlockRegionCount => _activeBlocks.Count;

		public int ActiveIndividualBlockCount { get { return _activeBlocks.Values.Sum(set => set.Count); } }

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

		private readonly Dictionary<int, ISet<RecordBlock>> _activeBlocks;
		private readonly Dictionary<int, ISet<WeakRecordBlock>> _cachedBlocks;
		private readonly ICacheProvider _cacheProvider;

		private int _lastReadRawDataBlockIndex = -1;
		private long _totalCachedBlockSizeBytes;
		private int _lastAvailableBlockIndex = int.MaxValue;
		private readonly ISet<IRecordExtractor> _recordExtractors;

		private readonly bool _autoSetBlockSize;
		private bool _autoSetExternalChangeBlockSize;

		private readonly Semaphore _availableBlocksSemaphore;

		/// <summary>
		/// Create a dataset with a certain unique name and the record extractors to use. 
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="recordExtractors">The record extractors to fetch the data from, which provide the dataset with ready to use record blocks.</param>
		public Dataset(string name, params IRecordExtractor[] recordExtractors) : this(name, BlockSizeAuto, recordExtractors)
		{
		}

		/// <summary>
		/// Create a dataset with a certain unique name, target block size in records and the record extractors to use.
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="blockSizeRecords">The target block size for records. May also be <see cref="BlockSizeAuto"/> or <see cref="BlockSizeAll"/>.</param>
		/// <param name="recordExtractors">The record extractors to fetch the data from, which provide the dataset with ready to use record blocks.</param>
		public Dataset(string name, int blockSizeRecords, params IRecordExtractor[] recordExtractors)
			: this(name, blockSizeRecords, new DiskCacheProvider(SigmaEnvironment.Globals.Get<string>("cache") + name), true, recordExtractors)
		{
		}

		/// <summary>
		/// Create a dataset with a certain unique name, target block size in records, specific cache provider and the record extractors to use.
		/// </summary>
		/// <param name="name">The unique dataset name.</param>
		/// <param name="blockSizeRecords">The target block size for records. May also be <see cref="BlockSizeAuto"/> or <see cref="BlockSizeAll"/>.</param>
		/// <param name="cacheProvider">The cache provider to use for caching record blocks and raw reader data.</param>
		/// <param name="flushCache">Indicate whether the cache provider should be flushed (cleared) before use. Only disable if block size and extractors used do not change (otherwise undefined behaviour).</param>
		/// <param name="recordExtractors">The record extractors to fetch the data from, which provide the dataset with ready to use record blocks.</param>
		public Dataset(string name, int blockSizeRecords, ICacheProvider cacheProvider, bool flushCache = true, params IRecordExtractor[] recordExtractors)
		{
			if (name == null)
			{
				throw new ArgumentNullException(nameof(name));
			}

			if (recordExtractors == null)
			{
				throw new ArgumentNullException(nameof(recordExtractors));
			}

			if (recordExtractors.Length == 0)
			{
				throw new ArgumentException("Datasets require at least one record extractor, but none were given.");
			}

			if (cacheProvider == null)
			{
				throw new ArgumentNullException(nameof(cacheProvider));
			}

			if (blockSizeRecords == BlockSizeAll)
			{
				//just set to maximum amount of records, extracting returns the maximum available anyway and we can't know the actual availability yet
				TargetBlockSizeRecords = int.MaxValue;
			}
			else if (blockSizeRecords == BlockSizeAuto)
			{
				//somewhat temporary guesstimate, should probably expose the individual parameters 
				const long estimatedRecordSizeBytes = 1024;
				const double memoryToConsume = 0.2f;
				const long optimalNumberBlocks = 8;
				const int maxBlockSizeRecords = 1024;
				long availableSystemMemory = SystemInformationUtils.GetAvailablePhysicalMemoryBytes();

				TargetBlockSizeRecords = Math.Min(maxBlockSizeRecords, (int) (availableSystemMemory * memoryToConsume / estimatedRecordSizeBytes / optimalNumberBlocks));

				_autoSetBlockSize = true;
			}
			else if (blockSizeRecords == 0 || blockSizeRecords < -2)
			{
				throw new ArgumentException($"Block size in records must be either BLOCK_SIZE_ALL, BLOCK_SIZE_AUTO or > 0, but given block size was {blockSizeRecords}.");
			}
			else
			{
				TargetBlockSizeRecords = blockSizeRecords;
			}

			Name = name;
			AnalyseExtractors(recordExtractors);

			_cacheProvider = cacheProvider;
			_recordExtractors = new HashSet<IRecordExtractor>(recordExtractors);

			_availableBlocksSemaphore = new Semaphore(MaxConcurrentActiveBlocks, MaxConcurrentActiveBlocks);

			_activeBlocks = new Dictionary<int, ISet<RecordBlock>>();
			_cachedBlocks = new Dictionary<int, ISet<WeakRecordBlock>>();

			if (flushCache)
			{
				_logger.Info($"Flushing all caches for dataset \"{Name}\" as flushCache flag was set...");

				InvalidateAndClearCaches();

				_logger.Info($"Done flushing all caches for dataset \"{Name}.\"");
			}
		}

		public IDataset[] SplitBlockwise(params int[] parts)
		{
			return SplitBlockwise(this, parts);
		}

		public IDataset[] SplitRecordwise(params double[] parts)
		{
			return SplitRecordwise(this, parts);
		}

		public bool TrySetBlockSize(int blockSizeRecords)
		{
			if (blockSizeRecords == TargetBlockSizeRecords)
			{
				//nothing to do here
				return true;
			}

			if (!_autoSetBlockSize)
			{
				_logger.Info($"Cannot change block size as block size was not set automatically (attempted to change block size to {blockSizeRecords}.");

				return false;
			}

			if (_activeBlocks.Count > 0 || _cachedBlocks.Count > 0)
			{
				_logger.Info($"Cannot change block size as {_activeBlocks.Count + _cachedBlocks.Count} blocks were already fetched and are active or cached.");

				return false;
			}

			if (_autoSetExternalChangeBlockSize && blockSizeRecords != TargetBlockSizeRecords)
			{
				_logger.Info($"Cannot change block size to {blockSizeRecords}, block size is incompatible with another external block size change request (other request: {TargetBlockSizeRecords})");

				return false;
			}

			_autoSetExternalChangeBlockSize = true;
			TargetBlockSizeRecords = blockSizeRecords;

			return true;
		}

		private void AnalyseExtractors(IEnumerable<IRecordExtractor> extractors)
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

			SectionNames = sectionNames.ToArray();
		}

		public int GetNumberOfLoadedInactiveCachedBlocks()
		{
			return _cachedBlocks.Values.SelectMany(blockSet => blockSet).Count(block => block.Loaded);
		}

		public bool CanFetchBlocksAfter(int blockIndex)
		{
			return blockIndex <= _lastAvailableBlockIndex;
		}

		public async Task<Dictionary<string, INDArray>> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			//TODO check if block even could be fetched to not waste thread resources if shouldWaitUntilAvailable is false anyway

			return await Task.Run(() => FetchBlock(blockIndex, handler, shouldWaitUntilAvailable));
		}

		public Dictionary<string, INDArray> FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true)
		{
			Dictionary<string, INDArray> block = FetchBlockConstrained(blockIndex, handler);

			//block could be fetched directly without violating any constraints, return successfully
			if (block != null)
			{
				if (block.Count == 0)
				{
					throw new InvalidOperationException("Fetched block did not contain any named elements (was empty; is the extractor output correct?).");
				}

				_availableBlocksSemaphore.WaitOne();

				RegisterActiveBlock(block, blockIndex, handler);

				return block;
			}
			else
			{
				if (shouldWaitUntilAvailable)
				{
					_logger.Info($"Could not directly load block with index {blockIndex} for handler {handler} and shouldWaitUntilAvailable flag is set to true, waiting for available space...");

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

			RecordBlock recordBlock = new RecordBlock(block, blockIndex, firstNamedBlock.Shape[0],
				handler.GetSizeBytes(block.Values.ToArray()), handler)
			{ Loaded = true, Active = true };

			lock (this)
			{
				TotalActiveBlockSizeBytes += recordBlock.EstimatedSizeBytes;
				TotalActiveRecords += recordBlock.NumberRecords;

				if (!_activeBlocks.ContainsKey(blockIndex))
				{
					_activeBlocks.Add(blockIndex, new HashSet<RecordBlock>());
				}

				_activeBlocks[blockIndex].Add(recordBlock);
			}
		}

		private void DeregisterActiveBlock(RecordBlock recordBlock)
		{
			if (!IsBlockActive(recordBlock.BlockIndex, recordBlock.Handler))
			{
				//block that should be de-registered is not even registered
				return;
			}

			lock (this)
			{
				TotalActiveBlockSizeBytes -= recordBlock.EstimatedSizeBytes;
				TotalActiveRecords -= recordBlock.NumberRecords;

				_activeBlocks[recordBlock.BlockIndex].Remove(recordBlock);

				if (_activeBlocks[recordBlock.BlockIndex].Count == 0)
				{
					_activeBlocks.Remove(recordBlock.BlockIndex);
				}
			}
		}

		private void RegisterCachedBlock(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler, bool keepReference)
		{
			if (IsBlockCached(blockIndex, handler))
			{
				//block's already cached, nothing to do here
				return;
			}

			if (!_cachedBlocks.ContainsKey(blockIndex))
			{
				_cachedBlocks.Add(blockIndex, new HashSet<WeakRecordBlock>());
			}

			WeakRecordBlock recordBlock = new WeakRecordBlock(keepReference ? block : null, blockIndex, block.First().Value.Shape[0], handler.GetSizeBytes(block.Values.ToArray()), handler);

			recordBlock.Loaded = false;

			_cachedBlocks[blockIndex].Add(recordBlock);
		}

		/// <summary>
		/// Invalidate and clear all caches associated with this dataset. 
		/// WARNING: Removing cache entries may cause certain datasets to load much more slowly or incorrectly. 
		///			 Use cases include removing cache entries for old datasets or datasets with different extractors. 
		/// </summary>
		public void InvalidateAndClearCaches()
		{
			_logger.Info("Invalidating and clearing all caches...");

			_cacheProvider.RemoveAll();

			_cachedBlocks.Clear();

			_logger.Info("Done invalidating and clearing all caches.");
		}

		private Dictionary<string, INDArray> FetchBlockWhenAvailable(int blockIndex, IComputationHandler handler)
		{
			while (true)
			{
				_availableBlocksSemaphore.WaitOne();

				_logger.Info($"Block region for request for block index {blockIndex} for handler {handler} became available, attempting to extract block to check if it fits all constraints...");

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
					if (blockIndex >= _lastAvailableBlockIndex)
					{
						return null;
					}
					else
					{
						_logger.Info($"Request for block with index {blockIndex} for handler {handler} was returned to the queue, waiting for available space...");

						_availableBlocksSemaphore.Release();
					}
				}
			}
		}

		private Dictionary<string, INDArray> FetchBlockConstrained(int blockIndex, IComputationHandler handler)
		{
			if (ActiveIndividualBlockCount >= MaxConcurrentActiveBlocks)
			{
				_logger.Info($"Unable to fetch block due to MaxConcurrentActiveBlocks constraint of {MaxConcurrentActiveBlocks}.");

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
				_logger.Info($"Unable to keep requested block {blockIndex} for handler {handler} in memory due to MaxTotalActiveBlockSizeBytes constraint of {MaxTotalActiveBlockSizeBytes} bytes (block of size {blockSizeBytes} would exceed constraint by {TotalActiveBlockSizeBytes + blockSizeBytes - MaxTotalActiveBlockSizeBytes} bytes.).");

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

			//check whether a block with the same index and format is already active 
			if (_activeBlocks.ContainsKey(blockIndex))
			{
				Dictionary<string, INDArray> block = GetBestMatchedConvertedBlock(_activeBlocks[blockIndex], handler);

				if (block != null)
				{
					return block;
				}
			}

			//check whether a block with the same index and format is already loaded and cached but not active
			if (_cachedBlocks.ContainsKey(blockIndex))
			{
				Dictionary<string, INDArray> block = GetBestMatchedConvertedBlock(_cachedBlocks[blockIndex], handler);

				if (block != null)
				{
					return block;
				}
			}

			lock (_cacheProvider)
			{
				string blockIdentifierInCache = $"extracted.{blockIndex}.{handler.DataType.Identifier}";

				//check whether a block of the same index and format is cached in the cache provider
				if (_cacheProvider.IsCached(blockIdentifierInCache))
				{
					Dictionary<string, INDArray> block = _cacheProvider.Load<Dictionary<string, INDArray>>(blockIdentifierInCache);

					//if its != null we could read it correctly in the right format
					if (block != null)
					{
						//register this cache entry as a properly loaded block in case the cache wasn't flushed and the cache map is outdated
						RegisterCachedBlock(block, blockIndex, handler, keepReference: false);

						return block;
					}
				}
			}

			return LoadAndExtractRaw(blockIndex, handler);
		}

		private Dictionary<string, INDArray> GetBestMatchedConvertedBlock(IEnumerable<RecordBlockBase> blocks, IComputationHandler handler)
		{
			RecordBlockBase bestMatchedBlock = null;

			foreach (RecordBlockBase otherBlock in blocks)
			{
				if (otherBlock.Loaded && handler.CanConvert(otherBlock.FirstNamedBlock, otherBlock.Handler))
				{
					if (handler.IsInterchangeable(otherBlock.Handler))
					{
						//no need to look any further, we already found the perfect match and can return without conversion
						return otherBlock.NamedBlockSections;
					}

					bestMatchedBlock = otherBlock;
				}
			}

			return bestMatchedBlock == null ? null : ConvertNamedBlocks(bestMatchedBlock.NamedBlockSections, handler);
		}

		private static Dictionary<string, INDArray> ConvertNamedBlocks(Dictionary<string, INDArray> namedBlockSections, IComputationHandler handler)
		{
			Dictionary<string, INDArray> convertedNamedBlocks = new Dictionary<string, INDArray>();

			foreach (string name in namedBlockSections.Keys)
			{
				convertedNamedBlocks.Add(name, handler.Convert(namedBlockSections[name], handler));
			}

			return convertedNamedBlocks;
		}

		private Dictionary<string, INDArray> LoadAndExtractRaw(int blockIndex, IComputationHandler handler)
		{
			// this cannot run concurrently as cache entries can only be read and written once without wasting resources and risking corruption of data
			lock (this)
			{
				if (blockIndex >= _lastReadRawDataBlockIndex)
				{
					object[] lastRawData = null;

					for (int tempBlockIndex = _lastReadRawDataBlockIndex + 1; tempBlockIndex <= blockIndex; tempBlockIndex++)
					{
						lastRawData = LoadDirect(tempBlockIndex, handler);

						//looks like we couldn't read any more blocks, maybe reached the end of the underlying source streams
						if (lastRawData == null)
						{
							return null;
						}

						if (AllowRawReadDataCaching)
						{
							_cacheProvider.Store($"raw.{tempBlockIndex}", lastRawData);
						}
					}

					return ExtractDirectFrom(lastRawData, blockIndex, handler);
				}
				else
				{
					if (AllowRawReadDataCaching)
					{
						string cacheIdentifier = $"raw.{blockIndex}";

						if (!_cacheProvider.IsCached(cacheIdentifier))
						{
							throw new InvalidOperationException($"Unable to load cached entry for block {blockIndex} for handler {handler}, cache entry does not exist in provider {_cacheProvider}.");
						}

						return ExtractDirectFrom(_cacheProvider.Load<object[]>(cacheIdentifier), blockIndex, handler);
					}
					else
					{
						throw new InvalidOperationException($"Cannot load and extract raw block with index {blockIndex} because AllowRawReadDataCaching is set to false and last read position is at {_lastReadRawDataBlockIndex}.");
					}
				}
			}
		}

		private object[] LoadDirect(int blockIndex, IComputationHandler handler)
		{
			IList<object> rawDataPerExtractor = new List<object>();

			PrepareExtractors();

			foreach (IRecordExtractor extractor in _recordExtractors)
			{
				object data;

				lock (extractor.Reader)
				{
					data = extractor.Reader.Read(TargetBlockSizeRecords);
				}

				//check if block reader could read anything, if not, return null
				if (data == null)
				{
					_lastAvailableBlockIndex = blockIndex - 1;

					_logger.Info($"Cannot load block {blockIndex} for handler {handler}, the underlying stream for extractor {extractor} is unable to retrieve any more records. End of stream most likely reached.");

					return null;
				}

				rawDataPerExtractor.Add(data);
			}

			if (blockIndex > _lastReadRawDataBlockIndex)
			{
				_lastReadRawDataBlockIndex = blockIndex;
			}

			return rawDataPerExtractor.ToArray();
		}

		private Dictionary<string, INDArray> ExtractDirectFrom(object[] data, int blockIndex, IComputationHandler handler)
		{
			Dictionary<string, INDArray> namedBlocks = new Dictionary<string, INDArray>();

			ITaskObserver prepareTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare, "extractors for dataset \"" + Name + "\"", indeterminate: true);

			PrepareExtractors();

			SigmaEnvironment.TaskManager.EndTask(prepareTask);

			ITaskObserver extractTask = SigmaEnvironment.TaskManager.BeginTask(TaskType.Prepare, $"block {blockIndex} for dataset \"{Name}\"", indeterminate: true);

			int extractorIndex = 0;
			foreach (IRecordExtractor extractor in _recordExtractors)
			{
				_logger.Info($"Extracting hierarchically from extractor {extractor} at index {extractorIndex}...");

				Dictionary<string, INDArray> subNamedBlock = extractor.ExtractHierarchicalFrom(data[extractorIndex++], TargetBlockSizeRecords, handler);

				//check if block size is 0, indicating we reached the end of the stream 
				if (subNamedBlock == null)
				{
					_lastAvailableBlockIndex = blockIndex - 1;

					_logger.Info($"Cannot  extract block {blockIndex} for handler {handler}, the underlying stream for extractor {extractor} is unable to retrieve any more records. End of stream most likely reached.");

					SigmaEnvironment.TaskManager.CancelTask(extractTask);

					return null;
				}

				foreach (string name in subNamedBlock.Keys)
				{
					if (namedBlocks.ContainsKey(name))
					{
						SigmaEnvironment.TaskManager.CancelTask(extractTask);

						throw new ArgumentException($"Section name collision: {name} is already used by another extractor, current extractor {extractor} cannot use it again.");
					}
					else
					{
						namedBlocks.Add(name, subNamedBlock[name]);
					}
				}
			}

			SigmaEnvironment.TaskManager.EndTask(extractTask);

			return namedBlocks;
		}

		public void FreeBlock(int blockIndex, IComputationHandler handler)
		{
			if (!_activeBlocks.ContainsKey(blockIndex))
			{
				_logger.Info($"Unable to free block with index {blockIndex} for handler {handler} because no block with that information is currently active.");

				return;
			}

			foreach (RecordBlock block in _activeBlocks[blockIndex])
			{
				if (ReferenceEquals(block.Handler, handler))
				{
					_logger.Info($"Freeing block with index {blockIndex} for handler {handler}...");

					CacheBlockConstrained(block.NamedBlockSections, blockIndex, handler);

					DeregisterActiveBlock(block);

					_availableBlocksSemaphore.Release();

					_logger.Info($"Done freeing block with index {blockIndex} for handler {handler}.");

					return;
				}
			}

			_logger.Info($"Unable to free block with index {blockIndex} for handler {handler} because no block with that information is currently active.");
		}

		private void CacheBlockConstrained(Dictionary<string, INDArray> block, int blockIndex, IComputationHandler handler)
		{
			if (_cachedBlocks.ContainsKey(blockIndex))
			{
				foreach (WeakRecordBlock cachedBlock in _cachedBlocks[blockIndex])
				{
					//check if block of the same type and size is already cached, if so, return, because there is no need to cache again
					if (cachedBlock.BlockIndex == blockIndex && cachedBlock.Handler.IsInterchangeable(handler) && block.First().Value.Shape[0] == cachedBlock.NumberRecords)
					{
						_logger.Info($"Skipping cache request of block {blockIndex} for handler {handler} because interchangeable block of same index, format and size is already cached.");

						return;
					}
				}
			}

			long blockSizeBytes = handler.GetSizeBytes(block.Values.ToArray());

			if (_cachedBlocks.Count >= MaxBlocksInCache)
			{
				_logger.Info($"Unable to cache block {blockIndex} for handler {handler} due to MaxBlocksInCache constraint of {MaxBlocksInCache}.");

				return;
			}

			if (blockSizeBytes + _totalCachedBlockSizeBytes >= MaxBytesInCache)
			{
				_logger.Info($"Unable to cache block {blockIndex} for handler {handler} due to MaxBytesInCache constraint of {MaxBytesInCache} bytes (block of size {blockSizeBytes} would exceed constraint by {_totalCachedBlockSizeBytes + blockSizeBytes - MaxBytesInCache} bytes).");

				return;
			}

			string cacheIdentifier = $"extracted.{blockIndex}.{handler.DataType.Identifier}";

			ITaskObserver task = SigmaEnvironment.TaskManager.BeginTask(TaskType.Save, cacheIdentifier, indeterminate: true);

			_cacheProvider.Store(cacheIdentifier, block);

			bool keepReference = TotalActiveBlockSizeBytes + blockSizeBytes < MaxTotalActiveBlockSizeBytes;

			RegisterCachedBlock(block, blockIndex, handler, keepReference);

			SigmaEnvironment.TaskManager.EndTask(task);

			_totalCachedBlockSizeBytes += blockSizeBytes;
		}

		private void PrepareExtractors()
		{
			foreach (IRecordExtractor extractor in _recordExtractors)
			{
				lock (extractor)
				{
					extractor.Prepare();
				}
			}
		}

		public long GetBlockSizeBytes(int blockIndex, IComputationHandler handler)
		{
			if (!_activeBlocks.ContainsKey(blockIndex))
			{
				return -1L;
			}

			foreach (RecordBlock block in _activeBlocks[blockIndex])
			{
				if (ReferenceEquals(block.Handler, handler))
				{
					return block.EstimatedSizeBytes;
				}
			}

			return -1L;
		}

		public bool IsBlockActive(int blockIndex)
		{
			return _activeBlocks.ContainsKey(blockIndex);
		}

		public bool IsBlockActive(int blockIndex, IComputationHandler handler)
		{
			if (!_activeBlocks.ContainsKey(blockIndex))
			{
				return false;
			}

			foreach (RecordBlock block in _activeBlocks[blockIndex])
			{
				if (ReferenceEquals(block.Handler, handler))
				{
					return true;
				}
			}

			return false;
		}

		private bool IsBlockCached(int blockIndex, IComputationHandler handler)
		{
			if (!_cachedBlocks.ContainsKey(blockIndex))
			{
				return false;
			}

			foreach (WeakRecordBlock block in _cachedBlocks[blockIndex])
			{
				if (ReferenceEquals(block.Handler, handler))
				{
					return true;
				}
			}

			return false;
		}

		public void Dispose()
		{
			foreach (IRecordExtractor extractor in _recordExtractors)
			{
				extractor.Dispose();
				extractor.Reader?.Dispose();
			}

			_cacheProvider.Dispose();
		}

		public static IDataset[] SplitBlockwise(IDataset dataset, params int[] parts)
		{
			if (parts.Length == 0)
			{
				throw new ArgumentException("Parts cannot be an empty collection.");
			}

			int splitInterval = parts.Sum();
			int lastEnd = 0;
			IDataset[] slices = new IDataset[parts.Length];

			for (int i = 0; i < parts.Length; i++)
			{
				slices[i] = new DatasetBlockwiseSlice(dataset, lastEnd, lastEnd + parts[i] - 1, splitInterval);
				lastEnd += parts[i];
			}

			return slices;
		}

		public static IDataset[] SplitRecordwise(IDataset dataset, params double[] parts)
		{
			if (parts.Length == 0)
			{
				throw new ArgumentException("Percentages cannot be an empty collection.");
			}

			if (parts.Sum() > 1.0)
			{
				throw new ArgumentException($"Percentages sum cannot be > 1.0, but parts sum was {parts.Sum()}.");
			}

			IDataset[] slices = new IDataset[parts.Length];

			double lastOffset = 0.0;

			for (int i = 0; i < slices.Length; i++)
			{
				slices[i] = new DatasetRecordwiseSlice(dataset, lastOffset, parts[i]);

				lastOffset += parts[i];
			}

			return slices;
		}

		internal abstract class RecordBlockBase
		{
			internal abstract Dictionary<string, INDArray> NamedBlockSections { get; set; }
			internal abstract INDArray FirstNamedBlock { get; set; }
			internal abstract bool Loaded { get; set; }

			internal IComputationHandler Handler;
			internal bool Active;
			internal int BlockIndex;
			internal long NumberRecords;
			internal long EstimatedSizeBytes;
		}

		internal class RecordBlock : RecordBlockBase
		{
			internal sealed override Dictionary<string, INDArray> NamedBlockSections { get; set; }
			internal sealed override INDArray FirstNamedBlock { get; set; }
			internal override bool Loaded { get; set; }

			public RecordBlock(Dictionary<string, INDArray> namedBlockSections, int blockIndex, long numberRecords, long estimatedSizeBytes, IComputationHandler handler)
			{
				NamedBlockSections = namedBlockSections;
				BlockIndex = blockIndex;
				NumberRecords = numberRecords;
				EstimatedSizeBytes = estimatedSizeBytes;
				Handler = handler;

				//record blocks internal block can be null
				if (namedBlockSections != null)
				{
					FirstNamedBlock = namedBlockSections[namedBlockSections.First().Key];
				}
			}
		}

		internal class WeakRecordBlock : RecordBlockBase
		{
			internal override Dictionary<string, INDArray> NamedBlockSections
			{
				get
				{
					Dictionary<string, INDArray> target;

					return _namedBlockSections.TryGetTarget(out target) ? target : null;
				}
				set
				{
					_namedBlockSections.SetTarget(value);
				}
			}

			internal override INDArray FirstNamedBlock
			{
				get
				{
					INDArray target;

					return _firstNamedBlock.TryGetTarget(out target) ? target : null;
				}
				set
				{
					_firstNamedBlock.SetTarget(value);
				}
			}

			internal override bool Loaded
			{
				get
				{
					Dictionary<string, INDArray> target;

					return _namedBlockSections.TryGetTarget(out target);
				}
				set
				{
				}
			}

			private readonly WeakReference<Dictionary<string, INDArray>> _namedBlockSections;
			private readonly WeakReference<INDArray> _firstNamedBlock;

			public WeakRecordBlock(Dictionary<string, INDArray> namedBlockSections, int blockIndex, long numberRecords, long estimatedSizeBytes, IComputationHandler handler)
			{
				_namedBlockSections = new WeakReference<Dictionary<string, INDArray>>(namedBlockSections);
				BlockIndex = blockIndex;
				NumberRecords = numberRecords;
				EstimatedSizeBytes = estimatedSizeBytes;
				Handler = handler;

				//record blocks internal block can be null
				if (namedBlockSections != null)
				{
					_firstNamedBlock = new WeakReference<INDArray>(namedBlockSections[namedBlockSections.First().Key]);
				}
			}
		}
	}
}
