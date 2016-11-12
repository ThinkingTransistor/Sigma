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

namespace Sigma.Core.Data.Extractors
{
	public class ValueMappingExtractor : BaseExtractor
	{
		public override string[] SectionNames { get; protected set; }

		private Dictionary<string, IList<Dictionary<object, object>>> namedValueMappingLists;
		private Dictionary<string, Dictionary<object, int>> namedMappingListIndices;
		private Dictionary<string, Dictionary<object, int>> namedMappingListSize;
		private Dictionary<string, Dictionary<object, object>> namedValueMappingsDirect;
			
		public override Dictionary<string, INDArray> ExtractDirect(int numberOfRecords, IComputationHandler handler)
		{
			return ExtractFrom(Reader.Read(numberOfRecords), numberOfRecords, handler);
		}

		public override Dictionary<string, INDArray> ExtractFrom(object readData, int numberOfRecords, IComputationHandler handler)
		{
			throw new NotImplementedException();
		}

		public override void Dispose()
		{
			throw new NotImplementedException();
		}
	}
}
