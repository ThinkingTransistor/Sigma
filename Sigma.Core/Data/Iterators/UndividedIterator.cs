/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System.Collections.Generic;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Iterators
{
	/// <summary>
	/// An undivided data iterator, which yields the entire available dataset as one block when yielded.
	/// Note: Undivided data iterators are extremely performance intensive for large datasets and may lead to systems running out of memory.
	/// </summary>
	public class UndividedIterator : BaseIterator
	{
		public UndividedIterator(IDataset dataset) : base(dataset)
		{
		}

		public override Dictionary<string, INDArray> Yield(IComputationHandler handler, SigmaEnvironment environment)
		{
			throw new System.NotImplementedException();
		}
	}
}
