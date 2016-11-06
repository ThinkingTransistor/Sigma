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

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			log4net.Config.XmlConfigurator.Configure();

			SigmaEnvironment.Globals["webProxy"] = new WebProxy();

			CSVRecordReader reader = new CSVRecordReader(new URLSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"));
			IRecordExtractor extractor = reader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");

			IComputationHandler handler = new CPUFloat32Handler();

			extractor.Prepare();

			var namedArrays = extractor.Extract(15, handler);

			foreach (string name in namedArrays.Keys)
			{
				//Console.WriteLine($"[{name}] =\n{string.Join(", ", namedArrays[name])}");
			}

			Console.ReadKey();
		}
	}
}
