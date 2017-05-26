using LiveCharts;
using LiveCharts.Geared;
using LiveCharts.Wpf;
using LiveCharts.Wpf.Charts.Base;
using Sigma.Core;
using Sigma.Core.Architecture;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Preprocessors.Adaptive;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Layers.Cost;
using Sigma.Core.Layers.External;
using Sigma.Core.Layers.Feedforward;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Monitors.WPF.Panels.Charts;
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Monitors.WPF.Panels.Parameterisation;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Monitors.WPF.View.Parameterisation;
using Sigma.Core.Persistence;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Processors;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Training.Optimisers.Gradient.Memory;
using Sigma.Core.Utils;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Media;
using MaterialDesignColors;
using Sigma.Core.Handlers.Backends.Debugging;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Monitors;
using Sigma.Core.Monitors.WPF.Utils.Defaults.MNIST;

namespace Sigma.Tests.Internals.WPF
{
	internal class Program
	{
		internal class DemoType
		{
			internal readonly string Name;
			internal readonly string Language;
			internal readonly bool Slow;
			internal readonly bool Dark;
			internal readonly Swatch PrimarySwatch;

			private readonly Func<SigmaEnvironment, ITrainer> _createTrainerFunction;

			public DemoType(string name, bool slow, Func<SigmaEnvironment, ITrainer> createTrainerFunction, Swatch primarySwatch, string language = "en-US", bool dark = true)
			{
				Name = name;
				Language = language;
				Slow = slow;
				_createTrainerFunction = createTrainerFunction;
				Dark = dark;
				PrimarySwatch = primarySwatch;
			}

			public DemoType(string name, bool slow, Func<SigmaEnvironment, ITrainer> createTrainerFunction, string language = "en-US", bool dark = true) : this(name, slow, createTrainerFunction, MaterialDesignValues.Teal, language, dark)
			{

			}

			internal ITrainer CreateTrainer(SigmaEnvironment sigma)
			{
				return _createTrainerFunction.Invoke(sigma);
			}

			internal static readonly DemoType Mnist = new DemoType("MNIST", true, CreateMnistTrainer);
			internal static readonly DemoType Iris = new DemoType("IRIS", false, CreateIrisTrainer);
			internal static readonly DemoType Xor = new DemoType("XOR", false, CreateXorTrainer);
			internal static readonly DemoType Wdbc = new DemoType("WDBC", false, CreateWdbcTrainer);
			internal static readonly DemoType Parkinsons = new DemoType("Parkinsons", false, CreateParkinsonsTrainer, MaterialDesignValues.LightBlue);
		}

		private static readonly DemoType DemoMode = DemoType.Parkinsons;

		private static void Main()
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("sigma_demo");

			// create a new mnist trainer
			string name = DemoMode.Name;
			ITrainer trainer = DemoMode.CreateTrainer(sigma);

			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_weights_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.StandardDeviation(a), "shared.network_weights_stddev"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_biases_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.StandardDeviation(a), "shared.network_biases_stddev"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.optimiser_updates_average"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.StandardDeviation(a), "shared.optimiser_updates_stddev"));
			trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*<external_output>._outputs.default.activations", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_activations_mean"));

			// create and attach a new UI framework
			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor(name, DemoMode.Language));
			gui.ColourManager.Dark = DemoMode.Dark;
			gui.ColourManager.PrimaryColor = DemoMode.PrimarySwatch;

			StatusBarLegendInfo iris = new StatusBarLegendInfo(name, MaterialColour.Blue);
			StatusBarLegendInfo general = new StatusBarLegendInfo("General", MaterialColour.Grey);
			gui.AddLegend(iris);
			gui.AddLegend(general);

			// create a tab
			gui.AddTabs("Overview", "Metrics", "Validation", "Maximisation", "Reproduction");

			// access the window inside the ui thread
			gui.WindowDispatcher(window =>
			{
				// enable initialisation
				window.IsInitializing = true;

				window.TabControl["Metrics"].GridSize = new GridSize(2, 4);
				window.TabControl["Validation"].GridSize = new GridSize(2, 5);
				window.TabControl["Maximisation"].GridSize = new GridSize(2, 5);
				window.TabControl["Reproduction"].GridSize = new GridSize(2, 5);

				window.TabControl["Overview"].GridSize.Rows -= 1;
				window.TabControl["Overview"].GridSize.Columns -= 1;

				// add a panel that controls the learning process
				window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control", trainer), legend: iris);

				ITimeStep reportTimeStep = DemoMode.Slow ? TimeStep.Every(1, TimeScale.Iteration) : TimeStep.Every(10, TimeScale.Epoch);

				var cost1 = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Cost / Epoch", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)).Linearify();
				var cost2 = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Cost / Epoch", trainer, "optimiser.cost_total", reportTimeStep);

				var weightAverage = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Weights / Epoch", trainer, "shared.network_weights_average", reportTimeStep, averageMode: true).Linearify();
				var weightStddev = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Weights / Epoch", trainer, "shared.network_weights_stddev", reportTimeStep, averageMode: true).Linearify();
				var biasesAverage = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Biases / Epoch", trainer, "shared.network_biases_average", reportTimeStep, averageMode: true).Linearify();
				var biasesStddev = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Biases / Epoch", trainer, "shared.network_biases_stddev", reportTimeStep, averageMode: true).Linearify();
				var updateAverage = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_average", reportTimeStep, averageMode: true).Linearify();
				var updateStddev = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Standard Deviation of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_stddev", reportTimeStep, averageMode: true).Linearify();

				var outputActivationsMean = CreateChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Mean of Output Activations", trainer, "shared.network_activations_mean", reportTimeStep, averageMode: true).Linearify();

				AccuracyPanel accuracy1 = null, accuracy2 = null;
				if (DemoMode != DemoType.Wdbc && DemoMode != DemoType.Parkinsons)
				{
					accuracy1 = new AccuracyPanel("Validation Accuracy", trainer, DemoMode.Slow ? TimeStep.Every(1, TimeScale.Epoch) : reportTimeStep, null, 1, 2);
					accuracy1.Fast().Linearify();
					accuracy2 = new AccuracyPanel("Validation Accuracy", trainer, DemoMode.Slow ? TimeStep.Every(1, TimeScale.Epoch) : reportTimeStep, null, 1, 2);
					accuracy2.Fast().Linearify();
				}

				IRegistry regTest = new Registry();
				regTest.Add("test", DateTime.Now);

				var parameter = new ParameterPanel("Parameters", sigma, window);
				parameter.Add("Time", typeof(DateTime), regTest, "test");

				ValueSourceReporter valueHook = new ValueSourceReporter(TimeStep.Every(1, TimeScale.Epoch), "optimiser.cost_total");
				trainer.AddGlobalHook(valueHook);
				sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);

				var costBlock = (UserControlParameterVisualiser)parameter.Content.Add("Cost", typeof(double), trainer.Operator.Registry, "optimiser.cost_total");
				costBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));

				var learningBlock = (UserControlParameterVisualiser)parameter.Content.Add("Learning rate", typeof(double), trainer.Operator.Registry, "optimiser.learning_rate");
				learningBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));

				window.TabControl["Overview"].AddCumulativePanel(cost1, 1, 2, legend: iris);
				window.TabControl["Overview"].AddCumulativePanel(parameter);
				//window.TabControl["Overview"].AddCumulativePanel(accuracy1, 1, 2, legend: iris);

				//window.TabControl["Metrics"].AddCumulativePanel(cost2, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(weightAverage, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(biasesAverage, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(updateAverage, legend: iris);
				if (accuracy2 != null)
				{
					window.TabControl["Metrics"].AddCumulativePanel(accuracy2, legend: iris);
				}

				window.TabControl["Metrics"].AddCumulativePanel(weightStddev, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(biasesStddev, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(updateStddev, legend: iris);
				window.TabControl["Metrics"].AddCumulativePanel(outputActivationsMean, legend: iris);

				if (DemoMode == DemoType.Mnist)
				{
					NumberPanel outputpanel = new NumberPanel("Numbers", trainer);
					DrawPanel drawPanel = new DrawPanel("Draw", trainer, 560, 560, 20, outputpanel);

					window.TabControl["Validation"].AddCumulativePanel(drawPanel, 2, 3);
					window.TabControl["Validation"].AddCumulativePanel(outputpanel, 2);

					for (int i = 0; i < 10; i++)
					{
						window.TabControl["Maximisation"].AddCumulativePanel(new MnistBitmapHookPanel($"Target Maximisation {i}", i, trainer, TimeStep.Every(1, TimeScale.Start)));
					}
				}
				//for (int i = 0; i < 10; i++)
				//{
				//	window.TabControl["Reproduction"].AddCumulativePanel(new MnistBitmapHookPanel($"Target Maximisation 7-{i}", 8, 28, 28, trainer, TimeStep.Every(1, TimeScale.Start)));
				//}

				window.IsInitializing = false;
			});

			if (DemoMode == DemoType.Mnist)
			{
				sigma.AddMonitor(new HttpMonitor("http://+:8080/sigma/"));
			}

			// the operators should not run instantly but when the user clicks play
			sigma.StartOperatorsOnRun = false;

			sigma.Prepare();

			sigma.Run();
		}

		private static ITrainer CreateXorTrainer(SigmaEnvironment sigma)
		{
			RawDataset dataset = new RawDataset("xor");
			dataset.AddRecords("inputs", new[] { 0, 0 }, new[] { 0, 1 }, new[] { 1, 0 }, new[] { 1, 1 });
			dataset.AddRecords("targets", new[] { 0 }, new[] { 1 }, new[] { 1 }, new[] { 0 });

			ITrainer trainer = sigma.CreateTrainer("xor-trainer");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(2) + FullyConnectedLayer.Construct(1) + OutputLayer.Construct(1) + SquaredDifferenceCostLayer.Construct();
			trainer.TrainingDataIterator = new UndividedIterator(dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Operator = new CpuSinglethreadedOperator();
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.01);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch), reportEpochIteration: true));
			trainer.AddLocalHook(new ValueReporter("network.layers.1-fullyconnected._outputs.default.activations", TimeStep.Every(1, TimeScale.Epoch)));

			return trainer;
		}

		private static ITrainer CreateIrisTrainer(SigmaEnvironment sigma)
		{
			var irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
			IRecordExtractor irisExtractor = irisReader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica")
				.Preprocess(new OneHotPreprocessor("targets", minValue: 0, maxValue: 2))
				.Preprocess(new AdaptiveNormalisingPreprocessor(minOutputValue: 0.0, maxOutputValue: 1.0))
				.Preprocess(new ShufflePreprocessor());

			IDataset dataset = new ExtractedDataset("iris", ExtractedDataset.BlockSizeAuto, false, irisExtractor);

			ITrainer trainer = sigma.CreateTrainer("xor-trainer");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(2) + FullyConnectedLayer.Construct(1) + OutputLayer.Construct(1) + SquaredDifferenceCostLayer.Construct();
			trainer.TrainingDataIterator = new UndividedIterator(dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Operator = new CpuSinglethreadedOperator();
			trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.01);

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch), averageValues: true));
			trainer.AddLocalHook(new ValueReporter("network.layers.1-fullyconnected._outputs.default.activations", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddLocalHook(new CurrentEpochIterationReporter(TimeStep.Every(5, TimeScale.Epoch)));

			return trainer;
		}

		private static ITrainer CreateWdbcTrainer(SigmaEnvironment sigma)
		{
			IDataset dataset = Defaults.Datasets.Wdbc();

			ITrainer trainer = sigma.CreateTrainer("wdbc-trainer");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(30)
								+ FullyConnectedLayer.Construct(42)
								+ FullyConnectedLayer.Construct(24)
								+ FullyConnectedLayer.Construct(1)
								+ OutputLayer.Construct(1)
								+ SquaredDifferenceCostLayer.Construct()
			};

			trainer.TrainingDataIterator = new MinibatchIterator(72, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.005);
			trainer.Operator = new CpuSinglethreadedOperator(new CpuFloat32Handler());

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new UniClassificationAccuracyReporter("validation", 0.5, TimeStep.Every(1, TimeScale.Epoch)));

			return trainer;
		}

		private static ITrainer CreateParkinsonsTrainer(SigmaEnvironment sigma)
		{
			IDataset dataset = Defaults.Datasets.Parkinsons();

			ITrainer trainer = sigma.CreateTrainer("parkinsons-trainer");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(22)
								+ FullyConnectedLayer.Construct(140)
								+ FullyConnectedLayer.Construct(20)
								+ FullyConnectedLayer.Construct(1)
								+ OutputLayer.Construct(1)
								+ SquaredDifferenceCostLayer.Construct()
			};

			trainer.TrainingDataIterator = new MinibatchIterator(10, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.01);
			trainer.Operator = new CpuSinglethreadedOperator(new DebugHandler(new CpuFloat32Handler()));

			trainer.AddInitialiser("*.*", new GaussianInitialiser(standardDeviation: 0.1));

			trainer.AddLocalHook(new AccumulatedValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new UniClassificationAccuracyReporter("validation", 0.5, TimeStep.Every(1, TimeScale.Epoch)));

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

			IDataset dataset = new ExtractedDataset("mnist", ExtractedDataset.BlockSizeAuto, false, mnistImageExtractor, mnistTargetExtractor);
			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(28, 28)
								+ FullyConnectedLayer.Construct(28 * 28)
								+ FullyConnectedLayer.Construct(10)
								+ OutputLayer.Construct(10)
								+ SoftMaxCrossEntropyCostLayer.Construct()
			};
			trainer.Network = Serialisation.ReadBinaryFileIfExists("mnist.sgnet", trainer.Network);

			trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(dataset));
			trainer.Optimiser = new MomentumGradientOptimiser(learningRate: 0.01, momentum: 0.9);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1f));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.1f, mean: 0.03f));

			//trainer.AddGlobalHook(new TargetMaximisationReporter(trainer.Operator.Handler.NDArray(ArrayUtils.OneHot(3, 10), 10L), TimeStep.Every(1, TimeScale.Start)));

			return trainer;
		}

		private static TrainerChartPanel<TChart, TSeries, TChartValues, TData> CreateChartPanel<TChart, TSeries, TChartValues, TData>(string title, ITrainer trainer, string hookedValue, ITimeStep timestep, bool averageMode = false) where TChart : Chart, new() where TSeries : Series, new() where TChartValues : IList<TData>, IChartValues, new()
		{
			var panel = new TrainerChartPanel<TChart, TSeries, TChartValues, TData>(title, trainer, hookedValue, timestep, averageMode);

			panel.Fast();
			panel.MaxPoints = 50;

			return panel;
		}
	}
}