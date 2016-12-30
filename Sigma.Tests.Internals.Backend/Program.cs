using System;
using System.Collections.Generic;
using System.Threading;
using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Layers;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.Training;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.Backend
{
	internal class Program
	{
		public static void Main(string[] args)
		{
			log4net.Config.XmlConfigurator.Configure();

			SigmaEnvironment.Globals["webProxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

			SampleNetworkArchitecture();

			Console.ReadKey();
		}

		private static void SampleNetworkArchitecture()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			IComputationHandler handler = new CpuFloat32Handler();
			ITrainer trainer = sigma.CreateTrainer("testtrainer");
			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(28, 28) +
										   2 * (ElementwiseLayer.Construct(2) + ElementwiseLayer.Construct(4)) +
										   OutputLayer.Construct(4);

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1f));
			trainer.AddInitialiser("*.bias", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));
			trainer.Initialise(handler);

			Console.WriteLine(trainer.Network.Registry);

			//foreach (ILayerBuffer buffer in trainer.Network.YieldLayerBuffersOrdered())
			//{
			//	Console.WriteLine(buffer.Layer.Name + ": ");

			//	Console.WriteLine("inputs:");
			//	foreach (string input in buffer.Inputs.Keys)
			//	{
			//		Console.WriteLine($"\t{input}: {buffer.Inputs[input].GetHashCode()}");
			//	}

			//	Console.WriteLine("outputs:");
			//	foreach (string output in buffer.Outputs.Keys)
			//	{
			//		Console.WriteLine($"\t{output}: {buffer.Outputs[output].GetHashCode()}");
			//	}
			//}
		}

		private static void SampleAutomaticDifferentiation()
		{
			IComputationHandler handler = new CpuFloat32Handler();

			uint traceTag = handler.BeginTrace();

			INDArray array = handler.NDArray(ArrayUtils.Range(1, 6), 2, 3);
			INumber a = handler.Number(-1.0f), b = handler.Number(3.0f);

			INumber c = handler.Trace(handler.Add(a, b), traceTag);
			INumber d = handler.Multiply(c, 2);
			INumber e = handler.Add(d, handler.Add(c, 3));
			INumber f = handler.Sqrt(e);

			array = handler.Multiply(array, f);

			INumber cost = handler.Sum(array);

			Console.WriteLine("cost: " + cost);

			handler.ComputeDerivativesTo(cost);

			Console.WriteLine(array);
			Console.WriteLine("f: " + handler.GetDerivative(f));
			Console.WriteLine("e: " + handler.GetDerivative(e));
			Console.WriteLine("d: " + handler.GetDerivative(d));
			Console.WriteLine("c: " + handler.GetDerivative(c));
			Console.WriteLine("a: " + handler.GetDerivative(array));

			handler.ComputeDerivativesTo(f);

			Console.WriteLine("f: " + handler.GetDerivative(f));
			Console.WriteLine("e: " + handler.GetDerivative(e));
			Console.WriteLine("d: " + handler.GetDerivative(d));
			Console.WriteLine("c: " + handler.GetDerivative(c));
			Console.WriteLine("a: " + handler.GetDerivative(array));
		}

		private static void SampleLoadExtractIterate()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			sigma.Prepare();

			//var irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
			//IRecordExtractor irisExtractor = irisReader.Extractor("inputs2", new[] { 0, 3 }, "targets2", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");
			//irisExtractor = irisExtractor.Preprocess(new OneHotPreprocessor(sectionName: "targets2", minValue: 0, maxValue: 2), new NormalisingPreprocessor(sectionNames: "inputs2", minInputValue: 0, maxInputValue: 6));

			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))));
			IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			ByteRecordReader mnistTargetReader = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1, source: new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))));
			IRecordExtractor mnistTargetExtractor = mnistTargetReader.Extractor("targets", new[] { 0L }, new[] { 1L }).Preprocess(new OneHotPreprocessor(minValue: 0, maxValue: 9));

			IComputationHandler handler = new CpuFloat32Handler();

			Dataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor, mnistTargetExtractor);
			IDataset[] slices = dataset.SplitRecordwise(0.8, 0.2);
			IDataset trainingData = slices[0];
			IDataset validationData = slices[1];

			MinibatchIterator trainingIterator = new MinibatchIterator(1, trainingData);
			MinibatchIterator validationIterator = new MinibatchIterator(1, validationData);

			while (true)
			{
				foreach (var block in trainingIterator.Yield(handler, sigma))
				{
					Thread.Sleep(100);

					PrintFormattedBlock(block);

					Thread.Sleep(1000);
				}
			}

			//Random random = new Random();
			//INDArray array = new ADNDArray<float>(3, 1, 2, 2);

			//new GaussianInitialiser(0.05, 0.05).Initialise(array, handler, random);

			//Console.WriteLine(array);

			//new ConstantValueInitialiser(1).Initialise(array, handler, random);

			//Console.WriteLine(array);

			//dataset.InvalidateAndClearCaches();
		}

		private static void PrintFormattedBlock(IDictionary<string, INDArray> block)
		{
			foreach (string name in block.Keys)
			{
				char[] palette = PrintUtils.AsciiGreyscalePalette;

				string blockString = name == "inputs"
					? ArrayUtils.ToString<float>(block[name], e => palette[(int) (e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false)
					: block[name].ToString();

				Console.WriteLine($"[{name}]=\n" + blockString);
			}
		}
	}
}
