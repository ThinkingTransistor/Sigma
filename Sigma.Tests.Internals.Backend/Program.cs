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
using Sigma.Core.Layers.Cost;
using Sigma.Core.Layers.External;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
using Sigma.Core.Persistence;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Hooks.Stoppers;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.Layers.Recurrent;
using Sigma.Core.Layers.Regularisation;
using Sigma.Core.Monitors;
using Sigma.Core.Training.Hooks.Saviors;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Training.Optimisers.Gradient.Memory;

namespace Sigma.Tests.Internals.Backend
{
	public static class Program
	{
		private static void Main(string[] args)
		{
			SigmaEnvironment.EnableLogging(xml: true);
			SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

			SampleMnist();

			Console.WriteLine("Program ended, waiting for termination, press any key...");
			Console.ReadKey();
		}

		private static void SampleHutter()
		{
			const long timeWindowSize = 50L;

			SigmaEnvironment sigma = SigmaEnvironment.Create("recurrent");

			IDataSource source = new MultiSource(new FileSource("enwik8"), new CompressedSource(new MultiSource(new FileSource("enwik8.zip"), new UrlSource("http://mattmahoney.net/dc/enwik8.zip"))));
			IRecordExtractor extractor = new CharacterRecordReader(source, (int)(timeWindowSize + 1), Encoding.ASCII)
				.Extractor(new ArrayRecordExtractor<short>(ArrayRecordExtractor<short>
					.ParseExtractorParameters("inputs", new[] { 0L }, new[] { timeWindowSize }, "targets", new[] { 0L }, new[] { timeWindowSize }))
					.Offset("targets", 1L))
				.Preprocess(new PermutePreprocessor(0, 2, 1))
				.Preprocess(new OneHotPreprocessor(0, 255));
			IDataset dataset = new ExtractedDataset("hutter", extractor);

			ITrainer trainer = sigma.CreateTrainer("hutter");

			trainer.Network.Architecture = InputLayer.Construct(256) + RecurrentLayer.Construct(256) + OutputLayer.Construct(256) + SoftMaxCrossEntropyCostLayer.Construct();
			trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
			trainer.AddNamedDataIterator("validation", new MinibatchIterator(100, dataset));
			trainer.Optimiser = new AdadeltaOptimiser(decayRate: 0.9);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.05));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Iteration), averageValues: true));
			trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(10, TimeScale.Iteration)));

			sigma.PrepareAndRun();
		}

		private static void SampleXor()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("logical");
			sigma.SetRandomSeed(0);
			sigma.Prepare();

			RawDataset dataset = new RawDataset("xor");
			dataset.AddRecords("inputs", new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 1 });
			dataset.AddRecords("targets", new[] { 0 }, new[] { 1 }, new[] { 1 }, new[] { 0 });

			ITrainer trainer = sigma.CreateTrainer("xor-trainer");

			trainer.Network.Architecture = InputLayer.Construct(2) + FullyConnectedLayer.Construct(2) + FullyConnectedLayer.Construct(1) + OutputLayer.Construct(1) + SquaredDifferenceCostLayer.Construct();
			trainer.TrainingDataIterator = new MinibatchIterator(1, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.1);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.05));

			trainer.AddLocalHook(new StopTrainingHook(atEpoch: 10000));
			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch), averageValues: true));
			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Stop), averageValues: true));
			trainer.AddLocalHook(new ValueReporter("network.layers.*<external_output>._outputs.default.activations", TimeStep.Every(1, TimeScale.Stop)));
			trainer.AddLocalHook(new ValueReporter("network.layers.*-fullyconnected.weights", TimeStep.Every(1, TimeScale.Stop)));
			trainer.AddLocalHook(new ValueReporter("network.layers.*-fullyconnected.biases", TimeStep.Every(1, TimeScale.Stop)));

			sigma.Run();
		}

		private static void SampleIris()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("iris");
			sigma.SetRandomSeed(0);

			sigma.Prepare();

			IDataset dataset = Defaults.Datasets.Iris();

			ITrainer trainer = sigma.CreateGhostTrainer("iris-trainer");

			trainer.Network.Architecture = InputLayer.Construct(4)
											+ FullyConnectedLayer.Construct(12)
											+ FullyConnectedLayer.Construct(3)
											+ OutputLayer.Construct(3)
											+ SquaredDifferenceCostLayer.Construct();
			//trainer.Network = Serialisation.ReadBinaryFileIfExists("iris.sgnet", trainer.Network);

			trainer.TrainingDataIterator = new MinibatchIterator(50, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.06);
			trainer.Operator = new CpuSinglethreadedOperator(new DebugHandler(new CpuFloat32Handler()));
			trainer.Operator.UseSessions = true;

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			//trainer.AddGlobalHook(new StopTrainingHook(atEpoch: 100));
			//trainer.AddLocalHook(new EarlyStopperHook("optimiser.cost_total", 20, target: ExtremaTarget.Min));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch), reportEpochIteration: true));
			//.On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));
			//trainer.AddLocalHook(new DiskSaviorHook<INetwork>("network.self", Namers.Dynamic("iris_epoch{0}.sgnet", "epoch"), verbose: true)
			//    .On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));

			trainer.AddHook(new MultiClassificationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: 1));
			trainer.AddHook(new StopTrainingHook(new ThresholdCriteria("shared.classification_accuracy_top1", ComparisonTarget.GreaterThanEquals, 0.98)));

			Serialisation.WriteBinaryFile(trainer, "trainer.sgtrainer");
			trainer = Serialisation.ReadBinaryFile<ITrainer>("trainer.sgtrainer");

			sigma.AddTrainer(trainer);

			sigma.AddMonitor(new HttpMonitor("http://+:8080/sigma/"));

			sigma.PrepareAndRun();
		}

		private static void SampleWdbc()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("wdbc");

			IDataset dataset = Defaults.Datasets.Wdbc();

			ITrainer trainer = sigma.CreateGhostTrainer("wdbc-trainer");

			trainer.Network.Architecture = InputLayer.Construct(30)
											+ FullyConnectedLayer.Construct(42)
											+ FullyConnectedLayer.Construct(24)
											+ FullyConnectedLayer.Construct(1)
											+ OutputLayer.Construct(1)
											+ SquaredDifferenceCostLayer.Construct();

			trainer.TrainingDataIterator = new MinibatchIterator(72, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.005);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new UniClassificationAccuracyReporter("validation", 0.5, TimeStep.Every(1, TimeScale.Epoch)));

			sigma.AddTrainer(trainer);

			sigma.AddMonitor(new HttpMonitor("http://+:8080/sigma/"));

			sigma.PrepareAndRun();
		}

		private static void SampleParkinsons()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("parkinsons");

			IDataset dataset = Defaults.Datasets.Parkinsons();

			ITrainer trainer = sigma.CreateGhostTrainer("parkinsons-trainer");

			trainer.Network.Architecture = InputLayer.Construct(22)
											+ FullyConnectedLayer.Construct(140)
											+ FullyConnectedLayer.Construct(20)
											+ FullyConnectedLayer.Construct(1)
											+ OutputLayer.Construct(1)
											+ SquaredDifferenceCostLayer.Construct();

			trainer.TrainingDataIterator = new MinibatchIterator(10, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.01);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new UniClassificationAccuracyReporter("validation", 0.5, TimeStep.Every(1, TimeScale.Epoch)));

			sigma.AddTrainer(trainer);

			sigma.PrepareAndRun();
		}

		private static void SampleMnist()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("mnist");

			sigma.Prepare();

			IDataset dataset = Defaults.Datasets.Mnist();

			ITrainer trainer = sigma.CreateTrainer("mnist-trainer");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(28, 28)
											+ DropoutLayer.Construct(0.2)
											+ FullyConnectedLayer.Construct(1000, activation: "rel")
											+ DropoutLayer.Construct(0.4)
											+ FullyConnectedLayer.Construct(800, activation: "rel")
											+ DropoutLayer.Construct(0.4)
											+ FullyConnectedLayer.Construct(10, activation: "sigmoid")
											+ OutputLayer.Construct(10)
											+ SoftMaxCrossEntropyCostLayer.Construct();
			//trainer.Network = Serialisation.ReadBinaryFileIfExists("mnist.sgnet", trainer.Network);
			trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			//trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.01);
			//trainer.Optimiser = new MomentumGradientOptimiser(learningRate: 0.01, momentum: 0.9);
			trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.016);
			trainer.Operator = new CpuSinglethreadedOperator();
			trainer.Operator.UseSessions = true;

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.05));

			//trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch), reportEpochIteration: true));
			trainer.AddLocalHook(new ValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Iteration), reportEpochIteration: true)
				.On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));

			//trainer.AddLocalHook(new DiskSaviorHook<INetwork>("network.self", Namers.Static("mnist_mincost.sgnet"), verbose: true)
			//	.On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));
			//trainer.AddGlobalHook(new DiskSaviorHook<INetwork>(TimeStep.Every(1, TimeScale.Epoch), "network.self", Namers.Static("mnist_maxacc.sgnet"), verbose: true)
			//	.On(new ExtremaCriteria("shared.classification_accuracy_top1", ExtremaTarget.Max)));

			var validationTimeStep = TimeStep.Every(1, TimeScale.Stop);

			//trainer.AddGlobalHook(new TargetMaximisationReporter(trainer.Operator.Handler.NDArray(ArrayUtils.OneHot(0, 10), 10L), TimeStep.Every(1, TimeScale.Start)));
			trainer.AddHook(new MultiClassificationAccuracyReporter("validation", validationTimeStep, tops: new[] { 1, 2, 3 }));
			//trainer.AddHook(new StopTrainingHook(new ThresholdCriteria("shared.classification_accuracy_top1", ComparisonTarget.GreaterThanEquals, 0.9), validationTimeStep));

			trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Iteration), 128));
			trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch), 4));
			trainer.AddHook(new StopTrainingHook(atEpoch: 10));

			sigma.Run();
		}

		private static void SampleTicTacToe()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("tictactoe");

			IDataset dataset = Defaults.Datasets.TicTacToe();

			ITrainer trainer = sigma.CreateTrainer("tictactoe-trainer");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(9)
											+ FullyConnectedLayer.Construct(63, "tanh")
											+ FullyConnectedLayer.Construct(90, "tanh")
											+ FullyConnectedLayer.Construct(3, "tanh")
											+ OutputLayer.Construct(3)
											+ SoftMaxCrossEntropyCostLayer.Construct();

			trainer.TrainingDataIterator = new MinibatchIterator(21, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new MomentumGradientOptimiser(learningRate: 0.01, momentum: 0.9);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new MultiClassificationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: new[] { 1, 2 }));

			trainer.AddGlobalHook(new DiskSaviorHook<INetwork>(TimeStep.Every(1, TimeScale.Epoch), "network.self", Namers.Static("tictactoe.sgnet"), verbose: true)
				.On(new ExtremaCriteria("shared.classification_accuracy_top1", ExtremaTarget.Max)));

			sigma.PrepareAndRun();
		}

		private static void SampleCachedFastIteration()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			IDataSource dataSource = new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")));

			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: dataSource);
			IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			IDataset dataset = new ExtractedDataset("mnist-training", ExtractedDataset.BlockSizeAuto, mnistImageExtractor);
			IDataset[] slices = dataset.SplitRecordwise(0.8, 0.2);
			IDataset trainingData = slices[0];

			Stopwatch stopwatch = Stopwatch.StartNew();

			IDataIterator iterator = new MinibatchIterator(10, trainingData);
			foreach (var block in iterator.Yield(new CpuFloat32Handler(), sigma))
			{
				//PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);
			}

			Console.Write("\nFirst iteration took " + stopwatch.Elapsed + "\n+=+ Iterating over dataset again +=+ Dramatic pause...");

			ArrayUtils.Range(1, 10).ToList().ForEach(i =>
			{
				Thread.Sleep(500);
				Console.Write(".");
			});

			stopwatch.Restart();

			foreach (var block in iterator.Yield(new CpuFloat32Handler(), sigma))
			{
				//PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);
			}

			Console.WriteLine("Second iteration took " + stopwatch.Elapsed);
		}

		private static void SampleDotProduct()
		{
			IComputationHandler handler = new CpuFloat32Handler();

			INDArray a = handler.NDArray(ArrayUtils.Range(1, 6), 3, 2);
			INDArray b = handler.NDArray(ArrayUtils.Range(1, 6), 2, 3);

			Console.WriteLine("a = " + ArrayUtils.ToString(a, (ADNDArray<float>.ToStringElement)null, 0, true));
			Console.WriteLine("b = " + ArrayUtils.ToString(b, (ADNDArray<float>.ToStringElement)null, 0, true));

			INDArray c = handler.Dot(a, b);

			Console.WriteLine("c = " + ArrayUtils.ToString(c, (ADNDArray<float>.ToStringElement)null, 0, true));
		}

		private static void SamplePermute()
		{
			IComputationHandler handler = new CpuFloat32Handler();

			INDArray array = handler.NDArray(ArrayUtils.Range(1, 30), 5L, 3L, 2L);

			Console.WriteLine(ArrayUtils.ToString(array, (ADNDArray<float>.ToStringElement)null, 0, true));

			array.PermuteSelf(1, 0, 2);

			Console.WriteLine(ArrayUtils.ToString(array, (ADNDArray<float>.ToStringElement)null, 0, true));
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
			trainer.Network = (INetwork)trainer.Network.DeepCopy();

			trainer.Operator = new CpuMultithreadedOperator(10);

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));
			trainer.Initialise(handler);

			trainer.Network = (INetwork)trainer.Network.DeepCopy();

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
			INumber f = handler.SquareRoot(e);

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

			ExtractedDataset dataset = new ExtractedDataset("mnist-training", ExtractedDataset.BlockSizeAuto, mnistImageExtractor, mnistTargetExtractor);
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

			//new GaussianInitialiser(0.05, 0.05).Initialise(array, Handler, random);

			//Console.WriteLine(array);

			//new ConstantValueInitialiser(1).Initialise(array, Handler, random);

			//Console.WriteLine(array);

			//dataset.InvalidateAndClearCaches();
		}

		private static void PrintFormatted(INDArray array, char[] palette)
		{
			string blockString = ArrayUtils.ToString<float>(array, e => palette[(int)(e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false);

			Console.WriteLine(blockString);
		}

		private static void PrintFormattedBlock(IDictionary<string, INDArray> block, char[] palette)
		{
			foreach (string name in block.Keys)
			{
				string blockString = ArrayUtils.ToString<float>(block[name], e => palette[(int)(e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false);

				Console.WriteLine($"[{name}]=\n" + blockString);
			}
		}
	}
}