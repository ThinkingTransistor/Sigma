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
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Training.Optimisers.Gradient.Memory;
using Sigma.Core.Utils;
using System;
using Sigma.Core.Monitors.WPF.Model.UI.Windows;
using Sigma.Core.Training.Modifiers;

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
            ITrainer trainer = CreateIrisTrainer(sigma);

            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_weights_average"));
            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.weights", (a, h) => h.StandardDeviation(a), "shared.network_weights_stddev"));
            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.network_biases_average"));
            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("network.layers.*.biases", (a, h) => h.StandardDeviation(a), "shared.network_biases_stddev"));
            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.Divide(h.Sum(a), a.Length), "shared.optimiser_updates_average"));
            trainer.AddLocalHook(new MetricProcessorHook<INDArray>("optimiser.updates", (a, h) => h.StandardDeviation(a), "shared.optimiser_updates_stddev"));

            // for the UI we have to activate more features
            if (UI)
            {
                // create and attach a new UI framework
                WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("MNIST"));

                StatusBarLegendInfo iris = new StatusBarLegendInfo("IRIS", MaterialColour.Blue);
                StatusBarLegendInfo general = new StatusBarLegendInfo("General", MaterialColour.Yellow);
                gui.AddLegend(iris);
                gui.AddLegend(general);

                // create a tab
                gui.AddTabs("Overview", "Metrics", "Validation");

                // access the window inside the ui thread
                gui.WindowDispatcher(window =>
                {
                    // enable initialisation
                    window.IsInitializing = true;

                    window.TabControl["Metrics"].GridSize = new GridSize(2, 4);
                    window.TabControl["Validation"].GridSize = new GridSize(1, 2);

                    window.TabControl["Overview"].GridSize.Rows -= 1;
                    window.TabControl["Overview"].GridSize.Columns -= 1;

                    // add a panel that controls the learning process
                    window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control", trainer), legend: iris);

                    ITimeStep reportTimeStep = TimeStep.Every(10, TimeScale.Epoch);

                    var cost = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Cost / Epoch", trainer, "optimiser.cost_total", reportTimeStep);
                    cost.Fast();

                    var weightAverage = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Mean of Weights / Epoch", trainer, "shared.network_weights_average", reportTimeStep, averageMode: true);
                    weightAverage.Fast();

                    var weightStddev = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Standard Deviation of Weights / Epoch", trainer, "shared.network_weights_stddev", reportTimeStep, averageMode: true);
                    weightStddev.Fast();

                    var biasesAverage = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Mean of Biases / Epoch", trainer, "shared.network_biases_average", reportTimeStep, averageMode: true);
                    biasesAverage.Fast();

                    var biasesStddev = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Standard Deviation of Biases / Epoch", trainer, "shared.network_biases_stddev", reportTimeStep, averageMode: true);
                    biasesStddev.Fast();

                    var updateAverage = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Mean of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_average", reportTimeStep, averageMode: true);
                    updateAverage.Fast();

                    var updateStddev = new TrainerChartPanel<CartesianChart, LineSeries, TickChartValues<double>, double>("Standard Deviation of Parameter Updates / Epoch", trainer, "shared.optimiser_updates_stddev", reportTimeStep, averageMode: true);
                    updateStddev.Fast();

                    var accuracy = new AccuracyPanel("Validation Accuracy", trainer, reportTimeStep, null, 1, 2);
                    accuracy.Fast();

                    IRegistry regTest = new Registry();
                    regTest.Add("test", DateTime.Now);

                    var parameter = new ParameterPanel("Parameters", sigma, window);
                    parameter.Add("Time", typeof(DateTime), regTest, "test");

                    ValueSourceReporterHook valueHook = new ValueSourceReporterHook(TimeStep.Every(1, TimeScale.Epoch), "optimiser.cost_total");
                    trainer.AddGlobalHook(valueHook);
                    sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);

                    valueHook = new ValueSourceReporterHook(TimeStep.Every(1, TimeScale.Iteration), "iteration");
                    trainer.AddLocalHook(valueHook);
                    sigma.SynchronisationHandler.AddSynchronisationSource(valueHook);

                    var costBlock = (UserControlParameterVisualiser)parameter.Content.Add("Cost", typeof(double), trainer.Operator.Registry, "optimiser.cost_total");
                    costBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));

                    var learningBlock = (UserControlParameterVisualiser)parameter.Content.Add("Learning rate", typeof(double), trainer.Operator.Registry, "optimiser.learning_rate");
                    learningBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));

                    //trainer.AddGlobalHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch)));

                    //var heeBlock = new SigmaTimeBlock();
                    //heeBlock.AutoPollValues(trainer, TimeStep.Every(1, TimeScale.Epoch));
                    //parameter.Content.Add(new Label { Content = "Cost" }, heeBlock, null, "optimiser.cost_total");

                    //window.TabControl["Overview"].AddCumulativePanel(cost, 1, 2, legend: iris);
                    window.TabControl["Overview"].AddCumulativePanel(parameter);
                    //window.TabControl["Overview"].AddCumulativePanel(accuracy, 1, 2, legend: iris);

                    window.TabControl["Metrics"].AddCumulativePanel(cost, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(weightAverage, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(biasesAverage, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(updateAverage, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(accuracy, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(weightStddev, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(biasesStddev, legend: iris);
                    window.TabControl["Metrics"].AddCumulativePanel(updateStddev, legend: iris);

                    //window.TabControl["Overview"].AddCumulativePanel(new LogDataGridPanel("Log"), 1, 3, general);

                    // finish initialisation
                    window.IsInitializing = false;
                });

                // the operators should not run instantly but when the user clicks play
                sigma.StartOperatorsOnRun = false;
            }

            sigma.Prepare();

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

            trainer.AddValueModifier("network.layers.*.weights", new ClipValueModifier());
            trainer.AddValueModifier("network.layers.*.bias*", new ClipValueModifier());

            trainer.AddHook(new ValueReporterHook("optimiser.cost_total", TimeStep.Every(1, TimeScale.Epoch)));
            trainer.AddHook(new ValidationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: 1));
            trainer.AddLocalHook(new CurrentEpochIterationReporter(TimeStep.Every(1, TimeScale.Epoch)));

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
