/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Architecture.Linear;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training;
using Sigma.Core.Training.Optimisers;
using Sigma.Tests.Layers;
using System;
using System.Collections.Generic;
using Sigma.Core.Training.Optimisers.Gradient;

namespace Sigma.Tests.Training
{
	internal class MockTrainer : Trainer
	{
		internal MockTrainer() : this("test-" + new Random().Next()) // never gonna happen, no collisions, otherwise we rerun the unit tests and hope
		{
		}

		protected MockTrainer(string name) : base(name)
		{
			Network = new Network("test");
			Network.Architecture = new LinearNetworkArchitecture(MockLayer.Construct());
			Optimiser = new GradientDescentOptimiser(0.0);
			IRecordExtractor extractor = new MockRecordExtractor();
			extractor.SectionNames = new[] {"targets", "inputs"};
			extractor.Reader = new MockRecordReader();
			Sigma = SigmaEnvironment.GetOrCreate("testificate-mocktrainer");
			TrainingDataIterator = new UndividedIterator(new ExtractedDataset("testificate", extractor));
		}

		internal class MockRecordReader : IRecordReader
		{
			public void Dispose()
			{
			}

			public IDataSource Source { get; }

			public IRecordExtractor Extractor(IRecordExtractor extractor)
			{
				throw new NotImplementedException();
			}

			public void Prepare()
			{
			}

			public object Read(int numberOfRecords)
			{
				return new object();
			}
		}

		internal class MockRecordExtractor : BaseExtractor
		{
			public override void Prepare()
			{
			}

			public override Dictionary<string, INDArray> ExtractDirectFrom(object readData, int numberOfRecords, IComputationHandler handler)
			{
				return ExtractDirect(numberOfRecords, handler);
			}

			public override Dictionary<string, INDArray> ExtractDirect(int numberOfRecords, IComputationHandler handler)
			{
				var block = new Dictionary<string, INDArray>();

				block["targets"] = handler.NDArray(new[] { 1, 2, 3, 4 }, 4L);
				block["inputs"] = handler.NDArray(new[] { 4, 3, 2, 1, -1, -2, -3, 2 }, 2L, 1L, 4L);

				return block;
			}

			public override Dictionary<string, INDArray> ExtractHierarchicalFrom(object readData, int numberOfRecords, IComputationHandler handler)
			{
				return ExtractDirect(numberOfRecords, handler);
			}

			public override void Dispose()
			{
			}
		}
	}
}
