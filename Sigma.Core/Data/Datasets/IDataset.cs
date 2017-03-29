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
	/// A dataset representing a collection of record blocks, where blocks can be 
	/// </summary>
	public interface IDataset : IDisposable
	{
		/// <summary>
		/// The name and identifier of this dataset.
		/// Dataset names should be globally unique and easily identifiable. 
		/// </summary>
		string Name { get; }

		/// <summary>
		/// Indicate if this dataset is an online dataset (meaning new data might be added during runtime).
		/// By default, this is assumed to be false, indicating a static dataset.
		/// Note: Data iterators and may perform certain optimisations for static datasets, so set this to false if possible.
		/// </summary>
		bool Online { get; set; }

		/// <summary>
		/// The preferred per block size in records.
		/// Note: Not every block must obey this request (e.g. the last black might very well be a different size).
		/// </summary>
		int TargetBlockSizeRecords { get; }

		/// <summary>
		/// The maximum number of concurrently active blocks. 
		/// </summary>
		int MaxConcurrentActiveBlocks { get; }

		/// <summary>
		/// The maximum total concurrently active block size in bytes.
		/// </summary>
		long MaxTotalActiveBlockSizeBytes { get; }

		/// <summary>
		/// The total size of all currently active record blocks in system memory in bytes.
		/// </summary>
		long TotalActiveBlockSizeBytes { get; }

		/// <summary>
		/// The maxmimum number of blocks to keep in the cache (inactive blocks are written to a cache, typically on disk, to be reloaded later).
		/// </summary>
		int MaxBlocksInCache { get; set; }

		/// <summary>
		/// The maxmimum number of bytes to keep in the cache (inactive blocks are written to a cache, typically on disk, to be reloaded later).
		/// </summary>
		long MaxBytesInCache { get; set; }

		/// <summary>
		/// The names for all sections present in this dataset (e.g. "inputs", "targets").
		/// </summary>
		string[] SectionNames { get; }

		/// <summary>
		/// A set of currently active and loaded record block indices. 
		/// </summary>
		IReadOnlyCollection<int> ActiveBlockIndices { get; }

		/// <summary>
		/// The number of currently active and loaded record blocks, with different block formats counting as different blocks. 
		/// </summary>
		int ActiveIndividualBlockCount { get; }

		/// <summary>
		/// The number of currently active and loaded record blocks, with different block formats of the same region counting as one active block index.
		/// </summary>
		int ActiveBlockRegionCount { get; }

		/// <summary>
		/// Split this dataset block-wise into certain parts.
		/// </summary>
		/// <param name="parts">The parts to split into.</param>
		/// <returns>The dataset slices split according to the given parts.</returns>
		IDataset[] SplitBlockwise(params int[] parts);

		/// <summary>
		/// Split the dataset record-wise into certain parts.
		/// </summary>
		/// <param name="percentages">The percentages to split into.</param>
		/// <returns>The dataset slices split according to the given percentages (0.0 - x - 1.0).</returns>
		IDataset[] SplitRecordwise(params double[] percentages);

		/// <summary>
		/// Attempt to set the block size to a new block size if compatible with the current set block size (e.g. if was auto set and this is the first request).
		/// </summary>
		/// <param name="blockSizeRecords"></param>
		/// <returns>A boolean indicating whether the block size could be set to the requested block size.</returns>
		bool TrySetBlockSize(int blockSizeRecords);

		/// <summary>
		/// Checks whether a certain block index is currently active and loaded in any format.
		/// </summary>
		/// <param name="blockIndex">The block index to check.</param>
		/// <returns>A boolean indicating if the given block index is active and loaded.</returns>
		bool IsBlockActive(int blockIndex);

		/// <summary>
		/// Checks whether a certain block index is currently active and loaded in a certain handler format.
		/// </summary>
		/// <param name="blockIndex">The block index to check.</param>
		/// <param name="handler">The handler of which the format should match.</param>
		/// <returns>A boolean indicating if the given block index is active and loaded in the given handler format.</returns>
		bool IsBlockActive(int blockIndex, IComputationHandler handler);

		/// <summary>
		/// Get the (estimated) size of a block in system memory with a certain index and handler format in bytes.
		/// </summary>
		/// <param name="blockIndex">The block index.</param>
		/// <param name="handler">The handler.</param>
		/// <returns>The size of the block with the given index and handler format in bytes.</returns>
		long GetBlockSizeBytes(int blockIndex, IComputationHandler handler);

		/// <summary>
		/// Check whether any more blocks can be fetched after a specified block index. 
		/// </summary>
		/// <param name="blockIndex">The block index after which to check for more blocks.</param>
		/// <returns></returns>
		bool CanFetchBlocksAfter(int blockIndex);

		/// <summary>
		/// Fetch named record block with a certain index for a certain computation handler. 
		/// Load, prepare and convert the requested block to the format required by a certain handler unless it was already fetched and is still active in that format.
		/// If the specified block can currently not be loaded due to memory constraints (as specified in <see cref="MaxConcurrentActiveBlocks"/> and <see cref="MaxTotalActiveBlockSizeBytes"/>):
		///		- If shouldWaitUntilAvailable flag is set: the calling thread will wait until the block becomes available, fetch the block and return it. 
		///		- If shouldWaitUntilAvailable flag is not set: null will be returned. 
		/// </summary>
		/// <param name="blockIndex">The index of the record block to request.</param>
		/// <param name="handler">The handler for which the block should be requested (specifies the block format).</param>
		/// <param name="shouldWaitUntilAvailable">Indicate if this method should wait for the specified block to become available or return null if it is not immediately available when called.</param>
		/// <returns>A block record representing the named blocks at the given block index in the format required by the given handler or null if shouldWaitUntilAvailable is set to false and the specified block is unavailable.</returns>
		IDictionary<string, INDArray> FetchBlock(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true);

		/// <summary>
		/// Fetch a record block with a certain index for a certain computation handler asynchronously. 
		/// Load, prepare and convert the requested block to the format required by a certain handler unless it was already fetched and is still active in that format.
		/// If the specified block can currently not be loaded due to memory constraints (as specified in <see cref="MaxConcurrentActiveBlocks"/> and <see cref="MaxTotalActiveBlockSizeBytes"/>):
		///		- If shouldWaitUntilAvailable flag is set: the task will asynchronously wait until the block becomes available, fetch the block and return it to the caller. 
		///		- If shouldWaitUntilAvailable flag is not set: null will be returned immediately. 
		/// </summary>
		/// <param name="blockIndex">The index of the record block to request.</param>
		/// <param name="handler">The handler for which the block should be requested (specifies the block format).</param>
		/// <param name="shouldWaitUntilAvailable">Indicate if this method should wait for the specified block to become available or return null if it is not immediately available when called.</param>
		/// <returns>A block record representing the block at the given block index in the format required by the given handler or null if shouldWaitUntilAvailable is set to false and the specified block is unavailable.</returns>
		Task<IDictionary<string, INDArray>> FetchBlockAsync(int blockIndex, IComputationHandler handler, bool shouldWaitUntilAvailable = true);

		/// <summary>
		/// Frees a record block with a certain index associated with the given handler.
		/// If all other references to that block index in other formats are freed, the entire block is unloaded (freed) and set to inactive.
		/// </summary>
		/// <param name="blockIndex">The block index to free of the given handler.</param>
		/// <param name="handler">The computation handler the block was originally requested with.</param>
		void FreeBlock(int blockIndex, IComputationHandler handler);
	}
}
