using System;
using Sigma.Core;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using Sigma.Core.Data.Sources;
using System.IO;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Extractors;
using System.Collections.Generic;
using Sigma.Core.Handlers.Backends;
using Sigma.Core.Handlers;
using System.Net;
using Sigma.Core.Data.Datasets;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
	{
			log4net.Config.XmlConfigurator.Configure();

			SigmaEnvironment.Globals["webProxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

			CSVRecordReader reader = new CSVRecordReader(new URLSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"));
			//CSVRecordReader reader = new CSVRecordReader(new FileSource("iris.data"));

			IRecordExtractor extractor = reader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");
			IComputationHandler handler = new CPUFloat32Handler();

			Dataset dataset = new Dataset("iris", 50, extractor);

			dataset.InvalidateAndClearCaches();

			Console.WriteLine("dataset: " + dataset.Name);
			Console.WriteLine("sections: " + ArrayUtils.ToString(dataset.SectionNames));

			Console.WriteLine("read block 0");
			Dictionary<string, INDArray> namedArrays = dataset.FetchBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("read block 0 (again)");
			namedArrays = dataset.FetchBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("free block 0");

			//freeing the block once, properly
			dataset.FreeBlock(0, handler);

			Console.WriteLine("free block 1");

			//doing it again for good measure
			dataset.FreeBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("read block 0 (again after freed)");
			namedArrays = dataset.FetchBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("read block 1");
			namedArrays = dataset.FetchBlock(1, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("read block 2");
			namedArrays = dataset.FetchBlock(2, handler);

			Console.WriteLine("free block 1");
			dataset.FreeBlock(1, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("fetch block 1 (again)");
			namedArrays = dataset.FetchBlock(1, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine("fetch block 3 (doesn't exist)");
			namedArrays = dataset.FetchBlock(3, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			Console.WriteLine(SystemInformationUtils.GetAvailablePhysicalMemoryBytes());

			Console.ReadKey();
		}
	}
}
