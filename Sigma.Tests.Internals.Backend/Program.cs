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

			CSVRecordReader reader = new CSVRecordReader(new MultiSource(new URLSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")));

			
			IComputationHandler handler = new CPUFloat32Handler();

			//Dataset dataset = new Dataset("mnist-training", 5, extractor);

			//var block = dataset.FetchBlock(0, handler);

			//foreach (string name in block.Keys)
			//{
			//	Console.WriteLine("[name] = " + block[name]);
			//}

			Console.ReadKey();
		}
	}
}
