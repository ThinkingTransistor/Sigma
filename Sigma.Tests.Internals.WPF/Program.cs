using LiveCharts.Wpf;
using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Layers.Cost;
using Sigma.Core.Layers.External;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Panels.Charts;
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers.Gradient.Memory;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	internal class Program
	{
		private const bool UI = true;

		private static void Main()
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma-MNIST");

			// create a new mnist trainer
			ITrainer trainer = CreateMnistTrainer(sigma);

			// for the UI we have to activate more features
			if (UI)
			{
				// create and attach a new UI framework
				WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("MNIST"));

				// create a tab
				gui.AddTabs("Overview");

				// access the window inside the ui thread
				gui.WindowDispatcher(window =>
				{
					// enable initialisation
					window.IsInitializing = true;

					// add a panel that controls the learning process
					window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control", trainer));

					// create an accuracy cost that updates every iteration
					var cost = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Cost", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Iteration));
					// improve the chart performance
					cost.Fast();

					// add the newly created panel
					window.TabControl["Overview"].AddCumulativePanel(cost);

					// finish initialisation
					window.IsInitializing = false;
				});

				// the operators should not run instantly but when the user clicks play
				sigma.StartOperatorsOnRun = false;
			}

			sigma.Prepare();

			sigma.Run();
		}

		/// <summary>
		/// Create a MNIST trainer (writing recognition) will be added to an environemnt.
		/// </summary>
		/// <param name="sigma">The sigma environemnt this trainer will be assigned to.</param>
		/// <returns>The newly created trainer.</returns>
		private static ITrainer CreateMnistTrainer(SigmaEnvironment sigma)
		{
			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))));
			IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			ByteRecordReader mnistTargetReader = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1, source: new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))));
			IRecordExtractor mnistTargetExtractor = mnistTargetReader.Extractor("targets", new[] { 0L }, new[] { 1L }).Preprocess(new OneHotPreprocessor(minValue: 0, maxValue: 9));

			IDataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor, mnistTargetExtractor);
			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(28, 28)
				+ 2 * FullyConnectedLayer.Construct(28 * 28)
				+ FullyConnectedLayer.Construct(10)
				+ OutputLayer.Construct(10)
				+ SoftMaxCrossEntropyCostLayer.Construct()
			};

			trainer.TrainingDataIterator = new MinibatchIterator(8, dataset);
			trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.02);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.05f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01f, mean: 0.03f));

			trainer.AddGlobalHook(new CurrentEpochIterationReporter(TimeStep.Every(1, TimeScale.Iteration)));

			return trainer;
		}
	}
}
