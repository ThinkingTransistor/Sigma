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
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Panels.Charts;
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Monitors.WPF.Panels.Parameterisation;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Monitors.WPF.View.Parameterisation;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Processors;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers.Gradient.Memory;
using Sigma.Core.Utils;
using System;
using LiveCharts.Geared;
using Sigma.Core.Data.Preprocessors.Adaptive;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;


namespace Sigma.Tests.Internals.WPF
{
	internal class Program
	{
		private const bool SampleMnist = true;

		private static void Main()
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("sigma_demo");

			// create a new mnist trainer
			string name = SampleMnist ? "MNIST" : "IRIS";
			string overview = SampleMnist ? "Überblick" : "Overview";
			string metrics = SampleMnist ? "Metriken" : "Metrics";
			string validation = SampleMnist ? "Validierung" : "Validation";

			ITrainer trainer = SampleMnist ? CreateMnistTrainer(sigma) : CreateIrisTrainer(sigma);

			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_weights_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.StandardDeviation(a), "shared.network_weights_stddev"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_biases_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.StandardDeviation(a), "shared.network_biases_stddev"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.optimiser_updates_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.StandardDeviation(a), "shared.optimiser_updates_stddev"));

			// create and attach a new UI framework
			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor(name, SampleMnist ? "de-DE" : "en-EN"));
			gui.ColourManager.Dark = SampleMnist;

			StatusBarLegendInfo iris = new StatusBarLegendInfo(name, MaterialColour.Blue);
			StatusBarLegendInfo general = new StatusBarLegendInfo("General", MaterialColour.Grey);
			gui.AddLegend(iris);
			gui.AddLegend(general);

			// create a tab
			gui.AddTabs(overview, metrics);

			if (SampleMnist)
			{
				gui.AddTabs(validation);
			}

			// access the window inside the ui thread
			gui.WindowDispatcher(window =>
			{
				// enable initialisation
				window.IsInitializing = true;

				window.TabControl[metrics].GridSize = new GridSize(2, 3);
				if (SampleMnist)
				{
					window.TabControl[validation].GridSize = new GridSize(1, 2);
				}

				window.TabControl[overview].GridSize.Rows -= 1;
				window.TabControl[overview].GridSize.Columns -= 1;

				// add a panel that controls the learning process
				window.TabControl[overview].AddCumulativePanel(new ControlPanel("Control", trainer), legend: iris);

				ITimeStep reportTimeStep = SampleMnist ? TimeStep.Every(1, TimeScale.Iteration) : TimeStep.Every(10, TimeScale.Epoch);
				const int maxPoints = 50;

				var cost1 = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Cost / Epoch", trainer, "optimiser.cost_total", reportTimeStep);
				cost1.Fast(); cost1.Linearify(); cost1.MaxPoints = maxPoints;
				var cost2 = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Cost / Epoch", trainer, "optimiser.cost_total", reportTimeStep);
				cost2.Fast();

				var weightAverage = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Weights / Epoch", trainer, "shared.network_weights_average", reportTimeStep, averageMode: true);
				weightAverage.Fast(); weightAverage.Linearify(); weightAverage.MaxPoints = maxPoints;

				var weightStddev = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Weights / Epoch", trainer, "shared.network_weights_stddev", reportTimeStep, averageMode: true);
				weightStddev.Fast(); weightStddev.Linearify(); weightStddev.MaxPoints = maxPoints;

				var biasesAverage = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Biases / Epoch", trainer, "shared.network_biases_average", reportTimeStep, averageMode: true);
				biasesAverage.Fast(); biasesAverage.Linearify(); biasesAverage.MaxPoints = maxPoints;

				var biasesStddev = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Biases / Epoch", trainer, "shared.network_biases_stddev", reportTimeStep, averageMode: true);
				biasesStddev.Fast(); biasesStddev.Linearify(); biasesStddev.MaxPoints = maxPoints;

				var updateAverage = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_average", reportTimeStep, averageMode: true);
				updateAverage.Fast(); updateAverage.Linearify(); updateAverage.MaxPoints = maxPoints;

				var updateStddev = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_stddev", reportTimeStep, averageMode: true);
				updateStddev.Fast(); updateStddev.Linearify(); updateStddev.MaxPoints = maxPoints;

				var accuracy1 = new AccuracyPanel("Validation Accuracy", trainer, SampleMnist ? TimeStep.Every(1, TimeScale.Epoch) : reportTimeStep, null, 1, 2);
				accuracy1.Fast(); accuracy1.Linearify(); accuracy1.MaxPoints = maxPoints;
				var accuracy2 = new AccuracyPanel("Validation Accuracy", trainer, SampleMnist ? TimeStep.Every(1, TimeScale.Epoch) : reportTimeStep, null, 1, 2);
				accuracy2.Fast(); accuracy2.Linearify(); accuracy2.MaxPoints = maxPoints;

				IRegistry regTest = new Registry();
				regTest.Add("test", DateTime.Now);

				var parameter = new ParameterPanel("Parameters", sigma, window);
				parameter.Add("Time", typeof(DateTime), regTest, "test");

				ValueSourceReporterHook valueHook = new ValueSourceReporterHook(TimeStep.Every(1, TimeScale.Epoch), "optimiser.cost_total");
				trainer.AddGlobalHook(valueHook);
				sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);

				var costBlock = (UserControlParameterVisualiser) parameter.Content.Add("Cost", typeof(object), trainer.Operator.Registry, "optimiser.cost_total");
				costBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));

				if (!SampleMnist)
				{
					var learningBlock = (UserControlParameterVisualiser) parameter.Content.Add("Learning rate", typeof(double), trainer.Operator.Registry, "optimiser.base_learning_rate");
					learningBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));
				}

				window.TabControl[overview].AddCumulativePanel(cost1, 1, 2, legend: iris);
				window.TabControl[overview].AddCumulativePanel(parameter);
				window.TabControl[overview].AddCumulativePanel(accuracy1, 1, 2, legend: iris);

				window.TabControl[metrics].AddCumulativePanel(weightAverage, legend: iris);
				window.TabControl[metrics].AddCumulativePanel(biasesAverage, legend: iris);
				window.TabControl[metrics].AddCumulativePanel(updateAverage, legend: iris);
				window.TabControl[metrics].AddCumulativePanel(weightStddev, legend: iris);
				window.TabControl[metrics].AddCumulativePanel(biasesStddev, legend: iris);
				window.TabControl[metrics].AddCumulativePanel(updateStddev, legend: iris);

				if (SampleMnist)
				{
					var outputPanel = new NumberPanel("Vorhersage", trainer);
					var drawPanel = new DrawPanel("Zeichenbox", trainer, 560, 560, 20, outputPanel);

					window.TabControl[validation].AddCumulativePanel(drawPanel);
					window.TabControl[validation].AddCumulativePanel(outputPanel);

				}

				// finish initialisation
				window.IsInitializing = false;
			});

			// the operators should not run instantly but when the user clicks play
			sigma.StartOperatorsOnRun = false;

			sigma.Prepare();

			sigma.Run();
		}

		private static ITrainer CreateIrisTrainer(SigmaEnvironment sigma)
		{
			var irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
			IRecordExtractor irisExtractor = irisReader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica")
				.Preprocess(new OneHotPreprocessor("targets", minValue: 0, maxValue: 2))
				.Preprocess(new AdaptiveNormalisingPreprocessor(minOutputValue: 0.0, maxOutputValue: 1.0))
				.Preprocess(new ShufflePreprocessor());

			IDataset dataset = new Dataset("iris", Dataset.BlockSizeAuto, false, irisExtractor);

			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(4)
										   + FullyConnectedLayer.Construct(4)
										   + FullyConnectedLayer.Construct(24)
										   + FullyConnectedLayer.Construct(3)
										   + OutputLayer.Construct(3)
										   + SoftMaxCrossEntropyCostLayer.Construct();

			trainer.TrainingDataIterator = new MinibatchIterator(10, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new AdagradOptimiser(0.01);
			trainer.Operator = new CpuSinglethreadedOperator(new DebugHandler(new CpuFloat32Handler()));

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.3));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.1));

			return trainer;
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

			IDataset dataset = new Dataset("mnist", Dataset.BlockSizeAuto, false, mnistImageExtractor, mnistTargetExtractor);
			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(28, 28)
				+ FullyConnectedLayer.Construct(28 * 28)
				+ FullyConnectedLayer.Construct(10)
				+ OutputLayer.Construct(10)
				+ SoftMaxCrossEntropyCostLayer.Construct()
			};

			trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new AdadeltaOptimiser(decayRate: 0.9);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.1f, mean: 0.03f));

			return trainer;
		}
	}
}
