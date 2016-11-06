/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Handlers;
using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Datasets
{
	/// <summary>
	/// A dataset representing a collection of record blocks, where blocks can be 
	/// </summary>
	public interface IDataset
	{
		/// <summary>
		/// The preferred per block size in records.
		/// Note: Not every block must obey this request. 
		/// </summary>
		long BlockSizeRecords { get; }

		/// <summary>
		/// The preferred per block size in bytes.
		/// Note: Not every block must obey this request.  
		/// </summary>
		long BlockSizeBytes { get; }

		/// <summary>
		/// The names for all sections present in this dataset (e.g. "inputs", "targets").
		/// </summary>
		string[] SectionNames { get; }

		/// <summary>
		/// A set of indices of already traversed record blocks (includes currently active blocks).
		/// </summary>
		ISet<int> TraversedBlockIndices { get; }

		/// <summary>
		/// A set of indices of the currently active and loaded record blocks. 
		/// </summary>
		ISet<int> ActiveBlockIndices { get; }

		/// <summary>
		/// Request a record block with a certain index. Load, prepare and convert the requested block to the format required by a certain handler unless it is already loaded in that format.
		/// </summary>
		/// <param name="blockIndex">The index of the record block to request.</param>
		/// <param name="handler"></param>
		/// <returns></returns>
		INDArray RequestBlock(int blockIndex, IComputationHandler handler);

		void FreeBlock(int blockIndex, IComputationHandler handler);
	}
}
