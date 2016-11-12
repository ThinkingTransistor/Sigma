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

			ByteRecordReader imageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new URLSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))));
			ByteRecordExtractor imageExtractor = imageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L });

			ByteRecordReader labelReader = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1, source: new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"), new URLSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))));
			ByteRecordExtractor labelExtractor = labelReader.Extractor("targets", new[] { 0L }, new[] { 1L });

			IComputationHandler handler = new CPUFloat32Handler();

			Dataset dataset = new Dataset("mnist-training", 5, imageExtractor, labelExtractor);

			var block = dataset.FetchBlock(0, handler);

			foreach (string name in block.Keys)
			{
				Console.WriteLine($"[{name}]=\n" + block[name]);
			}

			Console.ReadKey();
		}
	}
}
