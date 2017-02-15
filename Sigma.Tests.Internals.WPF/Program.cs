using System.Threading;
using LiveCharts.Wpf;
using log4net.Core;
using log4net.Filter;
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
using Sigma.Core.Monitors.WPF.Panels.Control;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
		public static MinibatchIterator TrainingIterator;

		private static void Main(string[] args)
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("WPF Monitor Demo"));
			gui.Priority = ThreadPriority.Highest;

			gui.AddTabs("Overview", "Log");

			gui.WindowDispatcher(window =>
			{
				window.IsInitializing = true;
			});

			sigma.Prepare();

			ITrainer trainer = CreateIrisTrainer(sigma);

			gui.WindowDispatcherAsync(window =>
			{
				ControlPanel control = new ControlPanel("Control", trainer);
				window.TabControl["Overview"].AddCumulativePanel(control, 2);

				TrainerChartPanel<CartesianChart, LineSeries, double> costChart = new TrainerChartPanel<CartesianChart, LineSeries, double>("Cost", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch));
				costChart.Series.PointGeometrySize = 0;
				costChart.Content.Hoverable = false;
				costChart.Content.DataTooltip = null;

				window.TabControl["Overview"].AddCumulativePanel(costChart, 2, 2);

				CartesianTestPanel chart = new CartesianTestPanel("Top accuracy of epoch", trainer);
				chart.AxisY.MinValue = 0;
				chart.AxisY.MaxValue = 1;
				chart.Series.PointGeometrySize = 0;

				//window.TabControl["Overview"].AddCumulativePanel(chart);

				window.TabControl["Log"].GridSize = new[] { 1, 1 };
				window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log", new LevelRangeFilter { LevelMin = Level.Info }));

				window.IsInitializing = false;
			});

			sigma.StartOperatorsOnRun = false;
			sigma.Run();
		}

		private static ITrainer CreateIrisTrainer(SigmaEnvironment sigma)
		{
			var irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
			IRecordExtractor irisExtractor = irisReader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");
			irisExtractor = irisExtractor.Preprocess(new OneHotPreprocessor(sectionName: "targets", minValue: 0, maxValue: 2));
			irisExtractor = irisExtractor.Preprocess(new PerIndexNormalisingPreprocessor(0, 1, "inputs", 0, 4.3, 7.9, 1, 2.0, 4.4, 2, 1.0, 6.9, 3, 0.1, 2.5));

			IDataset dataset = new Dataset("iris", Dataset.BlockSizeAuto, irisExtractor);
			IDataset trainingDataset = dataset;
			IDataset validationDataset = dataset;

			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network();
			trainer.Network.Architecture = InputLayer.Construct(4)
										   + 5 * FullyConnectedLayer.Construct(3)
										   + OutputLayer.Construct(3)
										   + SquaredDifferenceCostLayer.Construct();
			trainer.TrainingDataIterator = new MinibatchIterator(4, trainingDataset);
			trainer.AddNamedDataIterator("validation", new UndividedIterator(validationDataset));
			trainer.Optimiser = new GradientDescentOptimiser(learningRate: 0.002);
			trainer.Operator = new CpuSinglethreadedOperator();

			trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.4));
			trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.01, mean: 0.05));

			trainer.AddHook(new ValueReporterHook("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
			trainer.AddHook(new ValidationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: 1));
			trainer.AddLocalHook(new CurrentEpochIterationReporter(TimeStep.Every(1, TimeScale.Epoch)));

			return trainer;
		}
	}
}