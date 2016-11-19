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
using Sigma.Core.Data.Datasets;
using Sigma.Core.MathAbstract;

namespace Sigma.Core.Data.Iterators
{
	public class MinibatchIterator : IDataIterator
	{
		public IDataset UnderlyingDataset { get; }
		public INDArray Yield()
		{
			throw new NotImplementedException();
		}
	}
}
