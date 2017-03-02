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
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Panels.Charts;
using Sigma.Core.Monitors.WPF.Panels.Control;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Monitors.WPF.Utils;
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
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("JI-Demo", "de-DE"));
			gui.AddTabs("Tab1", "Tab2");

			ITrainer trainer = CreateIrisTrainer(sigma);

			gui.WindowDispatcher(window =>
			{
				window.TabControl["Tab1"].AddCumulativePanel(new ControlPanel("Steuerung", trainer));
			});

			gui.ColourManager.PrimaryColor = MaterialDesignValues.Blue;
			gui.ColourManager.SecondaryColor = MaterialDesignValues.Lime;

			gui.WindowDispatcher(window =>
			{
				window.TabControl["Tab1"].GridSize = new[] { 2, 2 };
				window.DefaultGridSize = new[] { 3, 3 };
			});

			StatusBarLegendInfo info = new StatusBarLegendInfo("Trainer 1", MaterialColour.Yellow);
			gui.AddLegend(info);

			gui.WindowDispatcher(window =>
			{
				var cost = new TrainerChartPanel<CartesianChart, LineSeries, double>("Fehler", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch));
				cost.Fast();
				window.TabControl["Tab2"].AddCumulativePanel(cost, legend: info);
				window.TabControl["Tab2"].AddCumulativePanel(new LogTextPanel("Anleitung") { Content = { IsReadOnly = false } }, 2, 2, info);
			});

			sigma.Prepare();

			sigma.StartOperatorsOnRun = false;
			sigma.Run();
		}

		//private static void Main(string[] args)
		//{
		//	SigmaEnvironment.EnableLogging();
		//	SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");
		//	SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

		//	WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("WPF Monitor Demo", "de-DE"));
		//	//gui.Priority = ThreadPriority.Highest;

		//	gui.AddTabs("Überblick", "Log");

		//	gui.WindowDispatcher(window =>
		//	{
		//		window.IsInitializing = true;
		//	});


		//	ITrainer trainer = CreateIrisTrainer(sigma);

		//	gui.ColourManager.Alternate = false;
		//	gui.ColourManager.Dark = true;
		//	gui.ColourManager.PrimaryColor = MaterialDesignValues.DeepOrange;
		//	gui.ColourManager.SecondaryColor = MaterialDesignValues.Amber;

		//	StatusBarLegendInfo legend = new StatusBarLegendInfo("Dropout", MaterialColour.Green);
		//	gui.AddLegend(legend);

		//	gui.WindowDispatcherAsync(window =>
		//	{
		//		ControlPanel control = new ControlPanel("Steuerung", trainer);

		//		TrainerChartPanel<CartesianChart, LineSeries, double> costChart = new TrainerChartPanel<CartesianChart, LineSeries, double>("Fehlerfunktion", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch));
		//		costChart.Fast();
		//		costChart.MaxPoints = 500;

		//		AccuracyPanel chart = new AccuracyPanel("Genauigkeit der Epoche", trainer, tops: new[] { 1, 2 });
		//		chart.Fast(hoverEnabled: true);

		//		window.TabControl["Überblick"].GridSize.Rows -= 1;

		//		window.TabControl["Überblick"].AddCumulativePanel(chart, legend: legend);
		//		window.TabControl["Überblick"].AddCumulativePanel(control, 2, legend: legend);
		//		window.TabControl["Überblick"].AddCumulativePanel(costChart, 2, 2, legend);


		//		//ChartPanel<CartesianChart, LineSeries, double> testCollection = new ChartPanel<CartesianChart, LineSeries, double>("Test collection");
		//		//testCollection.SeriesCollection.

		//		window.TabControl["Log"].GridSize = new[] { 1, 1 };
		//		window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log", new LevelRangeFilter { LevelMin = Level.Info }));

		//		window.IsInitializing = false;
		//	});
		//	sigma.Prepare();

		//	sigma.StartOperatorsOnRun = false;
		//	sigma.Run();
		//}

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
										   + FullyConnectedLayer.Construct(10)
										   + FullyConnectedLayer.Construct(20)
										   + FullyConnectedLayer.Construct(10)
										   + FullyConnectedLayer.Construct(3)
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