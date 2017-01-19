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
using Sigma.Core.Layers.Cost;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Panels.Control;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Training;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.Backend
{
	public class Program
	{
		public static MinibatchIterator TrainingIterator;

		private static void Main(string[] args)
		{
			SigmaEnvironment.EnableLogging();

			SampleTrainerOperatorWorker();

			Console.ReadKey();
		}

		private static void SampleTrainerOperatorWorker()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("trainer_test");

			sigma.Prepare();

			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))));
			IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			ByteRecordReader mnistTargetReader = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1, source: new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))));
			IRecordExtractor mnistTargetExtractor = mnistTargetReader.Extractor("targets", new[] { 0L }, new[] { 1L }).Preprocess(new OneHotPreprocessor(minValue: 0, maxValue: 9));

			IDataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor, mnistTargetExtractor);
			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(28, 28) + FullyConnectedLayer.Construct(10) + OutputLayer.Construct(10) + SoftMaxCrossEntropyCostLayer.Construct();
			trainer.TrainingDataIterator = new MinibatchIterator(8, dataset);
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.01);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.05f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));

			sigma.Run();
		}

		private static void SampleNetworkMerging()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("merge_test");

			ITrainer[] trainers = new ITrainer[3];
			int[] constantValues = { 2, 10, 70 };

			//INetworkMerger merger = new WeightedNetworkMerger(10d, 10d, 1d);
			INetworkMerger merger = new AverageNetworkMerger();
			IComputationHandler handler = new CpuFloat32Handler();

			for (int i = 0; i < trainers.Length; i++)
			{
				trainers[i] = sigma.CreateTrainer($"MergeTrainer{i}");
				trainers[i].Network = new Network($"{i}");
				trainers[i].Network.Architecture = InputLayer.Construct(2, 2) + ElementwiseLayer.Construct(2 * 2) + OutputLayer.Construct(2);

				trainers[i].AddInitialiser("*.weights", new ConstantValueInitialiser(constantValues[i]));

				trainers[i].Operator = new CpuMultithreadedOperator(5);
				trainers[i].Initialise(handler);
			}

			foreach (ITrainer trainer in trainers)
			{
				Console.WriteLine(trainer.Network.Registry);
			}

			merger.AddMergeEntry("layers.*.weights");
			merger.Merge(trainers[1].Network, trainers[2].Network, handler);

			Console.WriteLine("*******************");
			foreach (ITrainer trainer in trainers)
			{
				Console.WriteLine(trainer.Network.Registry);
			}
		}

		private static void SampleNetworkArchitecture()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			IComputationHandler handler = new CpuFloat32Handler();
			ITrainer trainer = sigma.CreateTrainer("test_trainer");
			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(2, 2) +
											ElementwiseLayer.Construct(2 * 2) +
											FullyConnectedLayer.Construct(2) +
											2 * (FullyConnectedLayer.Construct(4) + FullyConnectedLayer.Construct(2)) +
											OutputLayer.Construct(2);
			trainer.Network = (INetwork) trainer.Network.DeepCopy();

			trainer.Operator = new CpuMultithreadedOperator(10);

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));
			trainer.Initialise(handler);

			trainer.Network = (INetwork) trainer.Network.DeepCopy();

			Console.WriteLine(trainer.Network.Registry);

			IRegistryResolver resolver = new RegistryResolver(trainer.Network.Registry);

			Console.WriteLine("===============");
			object[] weights = resolver.ResolveGet<object>("layers.*.weights");
			Console.WriteLine(string.Join("\n", weights));
			Console.WriteLine("===============");



			//foreach (ILayerBuffer buffer in trainer.Network.YieldLayerBuffersOrdered())
			//{
			//      Console.WriteLine(buffer.Layer.Name + ": ");

			//      Console.WriteLine("inputs:");
			//      foreach (string input in buffer.Inputs.Keys)
			//      {
			//              Console.WriteLine($"\t{input}: {buffer.Inputs[input].GetHashCode()}");
			//      }

			//      Console.WriteLine("outputs:");
			//      foreach (string output in buffer.Outputs.Keys)
			//      {
			//              Console.WriteLine($"\t{output}: {buffer.Outputs[output].GetHashCode()}");
			//      }
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

			INumber cost = handler.Divide(handler.Sum(array), array.Length);

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

					PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);

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

		private static void PrintFormattedBlock(IDictionary<string, INDArray> block, char[] palette)
		{
			foreach (string name in block.Keys)
			{
				string blockString = name == "inputs"
						? ArrayUtils.ToString<float>(block[name], e => palette[(int) (e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false)
						: block[name].ToString();

				Console.WriteLine($"[{name}]=\n" + blockString);
			}
		}
	}
}