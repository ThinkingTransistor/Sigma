using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends;
using Sigma.Core.Math;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Sigma.Tests.Internals.Backend
{
	class Program
	{
		static void Main(string[] args)
		{
			log4net.Config.XmlConfigurator.Configure();

			SigmaEnvironment.Globals["webProxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

			CSVRecordReader reader = new CSVRecordReader(new MultiSource(new FileSource("iris.data"), new URLSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));

			IRecordExtractor extractor = reader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");
			IComputationHandler handler = new CPUFloat32Handler();

			Dataset dataset = new Dataset("iris", 5, extractor);

			var block = dataset.FetchBlock(0, handler);

			foreach (string name in block.Keys)
			{
				Console.WriteLine("[name] = " + block[name]);
			}

			Console.ReadKey();
		}

		private static async Task FetchAsync(Dataset dataset, IComputationHandler handler)
		{
			var block1 = dataset.FetchBlockAsync(0, handler);
			var block3 = dataset.FetchBlockAsync(2, handler);
			var block2 = dataset.FetchBlockAsync(1, handler);

			Dictionary<string, INDArray> namedArrays1 = await block1;
			Dictionary<string, INDArray> namedArrays2 = await block2;
			Dictionary<string, INDArray> namedArrays3 = await block3;
		}
	}
}
