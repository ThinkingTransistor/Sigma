using System.Threading;
using LiveCharts.Wpf;
using log4net;
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
using Sigma.Core.Monitors.WPF.Panels.Controls;
using Sigma.Core.Monitors.WPF.Panels.Parameterisation;
using Sigma.Core.Monitors.WPF.Utils;
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
		public static MinibatchIterator TrainingIterator;

		public static ILog Log = LogManager.GetLogger(typeof(Program));

		private static void Main()
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("Sigma-Demo"/*, "de-DE"*/));
			gui.AddTabs("Tab1", "Tab2");

			ITrainer trainer = CreateIrisTrainer(sigma);

			gui.WindowDispatcher(window =>
			{
				window.IsInitializing = true;
				window.TabControl["Tab1"].AddCumulativePanel(new ControlPanel("Steuerung", trainer));
			});

			gui.ColourManager.PrimaryColor = MaterialDesignValues.Blue;
			gui.ColourManager.SecondaryColor = MaterialDesignValues.Lime;

			gui.WindowDispatcher(window =>
			{
				window.TabControl["Tab1"].GridSize = new[] { 2, 2 };
				window.DefaultGridSize = new[] { 3, 3 };
			});

			StatusBarLegendInfo info = new StatusBarLegendInfo("CNN", MaterialColour.Yellow);
			var info2 = new StatusBarLegendInfo("");
			gui.AddLegend(info);


			IRegistry registry = new Registry
			{
				["boolean"] = true,
				["string"] = "huhu",
				["learning_rate"] = 0.0
			};
			registry.Add("object", sigma);


			gui.WindowDispatcher(window =>
			{
				var cost = new TrainerChartPanel<CartesianChart, LineSeries, double>("Fehler", trainer, "optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch));
				cost.Fast();
				window.TabControl["Tab2"].AddCumulativePanel(cost, legend: info);


				var parameterPanel = new ParameterPanel("Parameter", window.ParameterVisualiser, sigma.SynchronisationHandler);
				parameterPanel.Content.Add("Boolean Test", typeof(bool), registry, "boolean");
				parameterPanel.Content.Add("String test", typeof(string), registry, "string");
				parameterPanel.Content.Add("Object test", typeof(SigmaEnvironment), registry, "object");

				parameterPanel.Content.Add("Learning Rate", typeof(double), registry, "learning_rate");
				parameterPanel.Content.Add("Learning Rate", typeof(double), trainer.Operator.Registry, "optimiser.learning_rate");

				//parameterPanel.Content.Add(new Label { Content = "Awesome" }, typeof(bool));
				//parameterPanel.Content.Add(new Label { Content = "very very long text" }, typeof(bool));


				window.TabControl["Tab1"].AddCumulativePanel(parameterPanel);

				//window.TabControl["Tab2"].AddCumulativePanel(new LogTextPanel("Anleitung") { Content = { IsReadOnly = false } }, 2, 2, info);

				window.IsInitializing = false;
			});

			sigma.Prepare();

			sigma.StartOperatorsOnRun = false;
			sigma.RunAsync();

			//Log.Warn(registry);

			IRegistryResolver resolver = new RegistryResolver(trainer.Operator.Registry);

			while (true)
			{
				double[] vals = resolver.ResolveGet<double>("optimiser.learning_rate");
				if (vals != null && vals.Length > 0)
				{
					Log.Info($"optimiser.learning_rate: {vals[0]}");
				}
				//Log.Info(trainer.Operator.Registry);
				//Debug.WriteLine(registry["boolean"]);
				//Debug.WriteLine(registry["string"]);
				//Debug.WriteLine(registry["learning_rate"]);
				Thread.Sleep(1000);
			}
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