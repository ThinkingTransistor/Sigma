using log4net;
using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Preprocessors.Adaptive;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Layers.Cost;
using Sigma.Core.Layers.External;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.MathAbstract.Backends.SigmaDiff;
using Sigma.Core.Monitors.Synchronisation;
using Sigma.Core.Persistence;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Hooks.Stoppers;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Mergers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers.Gradient.Memory;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;

namespace Sigma.Tests.Internals.Backend
{
    public static class Program
    {
        public static MinibatchIterator TrainingIterator;

        private static void Main(string[] args)
        {
            SigmaEnvironment.EnableLogging(xml: true);
            SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

            SampleTrainerOperatorWorkerIris();

            Console.WriteLine("Program ended, waiting for termination, press any key...");
            Console.ReadKey();
        }

        private static void SampleTrainerOperatorWorkerIris()
        {
            SigmaEnvironment sigma = SigmaEnvironment.Create("trainer_test");

            sigma.Prepare();

            var irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
            IRecordExtractor irisExtractor = irisReader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica")
                                                        .Preprocess(new OneHotPreprocessor(sectionName: "targets", minValue: 0, maxValue: 2))
                                                        .Preprocess(new AdaptiveNormalisingPreprocessor(minOutputValue: 0.0, maxOutputValue: 1.0));

            IDataset dataset = new Dataset("iris", Dataset.BlockSizeAuto, irisExtractor);

            ITrainer trainer = sigma.CreateGhostTrainer("test");

            trainer.Network = new Network();
            trainer.Network.Architecture = InputLayer.Construct(4)
                                            + FullyConnectedLayer.Construct(12)
                                            + FullyConnectedLayer.Construct(10)
                                            + FullyConnectedLayer.Construct(3)
                                            + OutputLayer.Construct(3)
                                            + SquaredDifferenceCostLayer.Construct();
            trainer.TrainingDataIterator = new MinibatchIterator(4, dataset);
            trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
            trainer.Optimiser = new AdadeltaOptimiser(decayRate: 0.9);
            trainer.Operator = new CpuMultithreadedOperator(workerCount: 2);

            trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.2));
            trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.1, mean: 0.0));

            trainer.AddGlobalHook(new StopTrainingHook(atEpoch: 100));
            //trainer.AddLocalHook(new EarlyStopperHook("optimiser.cost_total", 20, target: ExtremaTarget.Min));
            trainer.AddHook(new ValueReporterHook("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
            trainer.AddHook(new ValidationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: 1));
            trainer.AddHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch)));

            //trainer.AddGlobalHook(new CurrentEpochIterationReporter(TimeStep.Every(1, TimeScale.Epoch)));

            Serialisation.WriteBinaryFile(trainer, "trainer.sgtrainer");
            trainer = Serialisation.ReadBinaryFile<ITrainer>("trainer.sgtrainer");

            sigma.AddTrainer(trainer);

            //trainer.Operator.InvokeCommand(new TestCommand(() => { throw new NotImplementedException(); }, "optimiser.learning_rate"));
            trainer.Operator.InvokeCommand(new SetValueCommand("optimiser.learning_rate", 0.02d, () => {/* finished */}));

            sigma.Run();
        }

        [Serializable]
        private class TestCommand : BaseCommand
        {
            private readonly ILog _log = LogManager.GetLogger(typeof(TestCommand));
            public TestCommand(Action onFinish = null, params string[] requiredRegistryEntries) : base(onFinish, requiredRegistryEntries)
            {
                _log.Info("Test command created");
            }

            /// <summary>
            /// Invoke this hook with a certain parameter registry if optional conditional criteria are satisfied.
            /// </summary>
            /// <param name="registry">The registry containing the required values for this hook's execution.</param>
            /// <param name="resolver">A helper resolver for complex registry entries (automatically cached).</param>
            public override void SubInvoke(IRegistry registry, IRegistryResolver resolver)
            {
                _log.Info("Test command invoked");
                //resolver.ResolveSet("optimiser.learning_rate", 10);
            }
        }

        private static void SampleTrainerOperatorWorkerMnist()
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
            trainer.Network.Architecture = InputLayer.Construct(28, 28)
                                            + FullyConnectedLayer.Construct(28 * 28)
                                            + FullyConnectedLayer.Construct(10)
                                            + OutputLayer.Construct(10)
                                            + SquaredDifferenceCostLayer.Construct();
            trainer.TrainingDataIterator = new MinibatchIterator(4, dataset);
            trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
            trainer.Optimiser = new AdadeltaOptimiser(decayRate: 0.9);
            trainer.Operator = new CpuSinglethreadedOperator(new DebugHandler(new CpuFloat32Handler(), enabled: false));

            trainer.AddInitialiser("*.weights", new XavierInitialiser(scale: 5));
            trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));
            trainer.AddHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch)));

            trainer.AddHook(new RunningTimeReporter(TimeStep.Every(5, TimeScale.Iteration)));
            trainer.AddGlobalHook(new CurrentEpochIterationReporter(TimeStep.Every(5, TimeScale.Iteration)));
            trainer.AddLocalHook(new ValueReporterHook("optimiser.cost_total", TimeStep.Every(5, TimeScale.Iteration)));
            trainer.AddGlobalHook(new ValidationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: new[] { 1 }));

            sigma.Run();
        }

        private static void SampleCachedFastIteration()
        {
            SigmaEnvironment sigma = SigmaEnvironment.Create("test");

            IDataSource dataSource = new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")));

            ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: dataSource);
            IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

            IDataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor);
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

            //new GaussianInitialiser(0.05, 0.05).Initialise(array, Handler, random);

            //Console.WriteLine(array);

            //new ConstantValueInitialiser(1).Initialise(array, Handler, random);

            //Console.WriteLine(array);

            //dataset.InvalidateAndClearCaches();
        }

        private static void PrintFormattedBlock(IDictionary<string, INDArray> block, char[] palette)
        {
            foreach (string name in block.Keys)
            {
                string blockString = name == "inputs"
                        ? ArrayUtils.ToString<float>(block[name], e => palette[(int)(e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false)
                        : block[name].ToString();

                Console.WriteLine($"[{name}]=\n" + blockString);
            }
        }
    }
}