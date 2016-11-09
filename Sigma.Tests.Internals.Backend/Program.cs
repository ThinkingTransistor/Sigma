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

			Dataset dataset = new Dataset("iris", extractor);

			Console.WriteLine("dataset: " + dataset.Name);
			Console.WriteLine("sections: " + ArrayUtils.ToString(dataset.SectionNames));

			Dictionary<string, INDArray> namedArrays = dataset.FetchBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);

			//Dictionary<string, INDArray> secondNamedArrays = dataset.FetchBlock(1, handler);

			//Console.WriteLine("second " + (secondNamedArrays == null));

			dataset.FreeBlock(0, handler);

			Console.WriteLine("TotalActiveRecords: " + dataset.TotalActiveRecords);
			Console.WriteLine("TotalActiveBlockSizeBytes: " + dataset.TotalActiveBlockSizeBytes);


			Console.ReadKey();
		}
	}
}
