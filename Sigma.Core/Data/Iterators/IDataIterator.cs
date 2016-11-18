/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core.Data.Datasets;
using Sigma.Core.MathAbstract;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// A data iterator which iterates over a dataset in blocks. 
	/// Note: The blocks yielded are in the BatchTimeFeatures format, where Batch and Time are one-dimensional.
	/// </summary>
	public interface IDataIterator
	{
		/// <summary>
		/// The dataset underlying this data iterator.
		/// </summary>
		IDataset UnderlyingDataset { get; }

		/// <summary>
		/// Yield a block from this data iterator.
		/// If no more blocks can be yielded, return null. 
		/// </summary>
		/// <returns>A block from the underlying dataset.</returns>
		INDArray Yield();
	}
}
