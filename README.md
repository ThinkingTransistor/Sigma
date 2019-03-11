# Sigma 
[![Build Status (Master)](https://img.shields.io/travis/ThinkingTransistor/Sigma/master.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma)
[![Build Status (Development)](https://img.shields.io/travis/ThinkingTransistor/Sigma/development.svg?style=flat-square)](https://travis-ci.org/ThinkingTransistor/Sigma/branches)
[![Nuget (PreRelease)](https://img.shields.io/nuget/vpre/Sigma.Core.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core)
[![Nuget (PreRelease WPF)](https://img.shields.io/nuget/vpre/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core.Monitors.WPF)
[![MyGet (PreRelease)](https://img.shields.io/myget/sigma/v/Sigma.Core.svg?style=flat-square)](https://www.myget.org/feed/sigma/package/nuget/Sigma.Core)
[![MyGet (PreRelease WPF)](https://img.shields.io/myget/sigma/v/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.myget.org/feed/sigma/package/nuget/Sigma.Core.Monitors.WPF)
[![MIT license](https://img.shields.io/github/license/mashape/apistatus.svg?style=flat-square)](http://choosealicense.com/licenses/mit)

Rocket powered machine learning. Create, compare, adapt, improve - neural networks at the speed of thought.

Sigma was created to be a more plain and understandable machine learning framework. This is what it can do:

* Input, Output, Dense, Dropout, Recurrent, SoftmaxCE / SquaredDiff cost layers
* Gradient descent, Momentum, Adadelta, Adagrad optimisers
* Hooks for storing / restoring checkpoints, timekeeping, stopping (or doing other things) on certain criteria, computing and reporting runtime metrics
* Easy addition of new layers with functional automatic differentiation
* Linear and non-linear networks with arbitrarily connected constructs
* Distributed multi- and single- CPU and GPU (CUDA) backends 
* Native graphical interface where parameters can be set and monitored in real-time


## Installation

### NuGet [Recommended]

The recommended way to use the latest version of Sigma is adding the NuGet package to your project. 
You can either include the core framework (command line only) [![Nuget (PreRelease)](https://img.shields.io/nuget/vpre/Sigma.Core.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core) or the WPF visualiser (only works on Windows) which also references the core framework [![Nuget (PreRelease WPF)](https://img.shields.io/nuget/vpre/Sigma.Core.Monitors.WPF.svg?style=flat-square)](https://www.nuget.org/packages/Sigma.Core.Monitors.WPF). 

In both cases, you can use any project with a main (e.g. ConsoleApplication) but you have to change the project settings to x64 (since **Sigma only supports 64bit mode**) and change the target framework to **.NET 4.6 before** installing the NuGet packages. 

#### Change .NET Version (Visual Studio)
Right click the project in the solution explorer and click **Properties**. Then in the tab **Application**, change the **Target framework** to .NET 4.6. 

#### Change to x64 (Visual Studio)
In the navigation bar, click on **Any CPU** and select **Configuration Manager**. In the configuration manager, click the **Platform dropdown** and choose **new**. Change the **New platform** to **x64** and click **OK** and **close**.

### From source

For extensive customisation you can also install Sigma from source. This is not recommended as it may be outdated and unstable, but you still might want to do it for whatever reason. First, clone from the GitHub repository - use master for stable releases, development for recent and possibly unstable changes and fixes:

```
git clone https://github.com/ThinkingTransistor/Sigma
```

Restore and add all used NuGet packages (also see Used libraries) in the project folder (Sigma by default):

```
cd Sigma
nuget restore Sigma.sln
```

You can then integrate Sigma directly into your program as a project reference.

## First program - Classic MNIST (Sigma.Core)
An example trainer to classify handwritten digits from the MNIST dataset, using dense (fullyconnected) and dropout layers with a Softmax / Cross Entropy cost layer. Using a GTX 1080 and the CUDA backend, this simple trainer reaches 94.7% test accuracy after ~4 seconds (1 epoch), 97.0% after ~25 seconds (5 epochs + test time) and peaks at 97.6% after ~50 seconds (10 epochs + test time). 

For this program, the Sigma.Core NUGET package is requird.

```
SigmaEnvironment.EnableLogging();
SigmaEnvironment sigma = SigmaEnvironment.Create("mnist");
IDataset dataset = Defaults.Datasets.Mnist(); // datasets are automatically downloaded and unpacked if not on disk
ITrainer trainer = sigma.CreateTrainer("mnist-trainer");

trainer.Network.Architecture = InputLayer.Construct(28, 28)
				+ DropoutLayer.Construct(0.2)
				+ FullyConnectedLayer.Construct(1000, activation: "rel")
				+ DropoutLayer.Construct(0.4)
				+ FullyConnectedLayer.Construct(800, activation: "rel")
				+ DropoutLayer.Construct(0.4)
				+ FullyConnectedLayer.Construct(10, activation: "sigmoid")
				+ OutputLayer.Construct(10)
				+ SoftMaxCrossEntropyCostLayer.Construct();
trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
trainer.AddNamedDataIterator("validation", new UndividedIterator(Defaults.Datasets.MnistValidation()));
trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.02);

trainer.Operator = new CpuSinglethreadedOperator(); // change this line to a new CudaSinglethreadedOperator() if you have an NVIDIA GPU
// Of course, CUDA has to be installed in order to work.

trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1));
trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.05));

trainer.AddLocalHook(new ValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Iteration), reportEpochIteration: true)
	.On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));

trainer.AddHook(new MultiClassificationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: new[] { 1, 2, 3 }));

trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(10, TimeScale.Iteration), averageSpan: 32));
trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch), averageSpan: 4));
trainer.AddGlobalHook(new StopTrainingHook(atEpoch: 10));

sigma.PrepareAndRun();
```

## First program - Classic MNIST (Sigma.Core.WPF)
This is the same program as above, only with visualisation activated.

For this program, the Sigma.Core.WPF NUGET package is requird.

```
SigmaEnvironment.EnableLogging();
SigmaEnvironment sigma = SigmaEnvironment.Create("mnist");
IDataset dataset = Defaults.Datasets.Mnist(); // datasets are automatically downloaded and unpacked if not on disk
ITrainer trainer = sigma.CreateTrainer("mnist-trainer");

trainer.Network.Architecture = InputLayer.Construct(28, 28)
								+ DropoutLayer.Construct(0.2)
								+ FullyConnectedLayer.Construct(1000, activation: "rel")
								+ DropoutLayer.Construct(0.4)
								+ FullyConnectedLayer.Construct(800, activation: "rel")
								+ DropoutLayer.Construct(0.4)
								+ FullyConnectedLayer.Construct(10, activation: "sigmoid")
								+ OutputLayer.Construct(10)
								+ SoftMaxCrossEntropyCostLayer.Construct();
trainer.TrainingDataIterator = new MinibatchIterator(100, dataset);
trainer.AddNamedDataIterator("validation", new UndividedIterator(Defaults.Datasets.MnistValidation()));
trainer.Optimiser = new AdagradOptimiser(baseLearningRate: 0.02);

trainer.Operator = new CpuSinglethreadedOperator(); // change this line to a new CudaSinglethreadedOperator() if you have an NVIDIA GPU
													// Of course, CUDA has to be installed in order to work.


trainer.AddInitialiser("*.weights", new GaussianInitialiser(standardDeviation: 0.1));
trainer.AddInitialiser("*.bias*", new GaussianInitialiser(standardDeviation: 0.05));

trainer.AddLocalHook(new ValueReporter("optimiser.cost_total", TimeStep.Every(1, TimeScale.Iteration), reportEpochIteration: true)
	.On(new ExtremaCriteria("optimiser.cost_total", ExtremaTarget.Min)));

trainer.AddHook(new MultiClassificationAccuracyReporter("validation", TimeStep.Every(1, TimeScale.Epoch), tops: new[] { 1, 2, 3 }));

trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(10, TimeScale.Iteration), averageSpan: 32));
trainer.AddLocalHook(new RunningTimeReporter(TimeStep.Every(1, TimeScale.Epoch), averageSpan: 4));

WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("My first Sigma application"));
gui.AddTabs("Overview", "Validation");
gui.ColourManager.Dark = true;

gui.WindowDispatcher(window =>
{
	window.TabControl["Overview"].GridSize = new GridSize(2, 3);
	window.TabControl["Overview"].AddPanel(new ControlPanel("Control", trainer), 0, 0, 2);

	var cost = new TrainerChartPanel<CartesianChart, GLineSeries, GearedValues<double>, double>("Cost / Epoch", trainer,
		"optimiser.cost_total", TimeStep.Every(25, TimeScale.Iteration)).Fast().Linearify();
	cost.MaxPoints = 50;

	var accuracy = new AccuracyPanel("Validation Accuracy", trainer, TimeStep.Every(1, TimeScale.Epoch), null, 1, 2);

	window.TabControl["Overview"].AddPanel(cost, 0, 1, 1, 2);
	window.TabControl["Overview"].AddPanel(accuracy, 1, 1, 1, 2);

	NumberPanel outputpanel = new NumberPanel("Numbers", trainer);
	DrawPanel drawPanel = new DrawPanel("Draw", trainer, 560, 560, 20, outputpanel);

	window.TabControl["Validation"].GridSize = new GridSize(2, 4);
	window.TabControl["Validation"].AddCumulativePanel(drawPanel, 2, 3);
	window.TabControl["Validation"].AddCumulativePanel(outputpanel, 2);
});

// the operators should not run instantly but when the user clicks play
sigma.StartOperatorsOnRun = false;

sigma.PrepareAndRun();
```

### Output
![Example output of overview panel](https://i.imgur.com/g9Qxwhk.png)
![Example output of validation panel](https://i.imgur.com/Kd1q8WI.png)


## Documentation - how do I? 
The API-Documentation (of the master-branch) is always available at our [Github-Page](https://thinkingtransistor.github.io/Sigma/). If you want it locally available, clone the gh-pages branch.

## Acknowledgements 

The completion of this project would not have been possible without the assistance and support of many generous people. We cannot express enough thanks and gratefully acknowledge their contributions. In particular, we would like to express our deep gratitude and appreciation to the following: 

- Prof. Dr. Patrick van der Smagt, thank you. Thank you for your continued support and never-ending assistance. Thank you for your heartfelt encouragement and inspirational enthusiasm. Thank you for helping us out at midnight on a Saturday evening---we deeply appreciate your time, kindness, and efforts as our advisor. 

- To our family, friends, and loved ones, we thank you for your support and gratefully acknowledge your assistance in making this project become a reality. 

- To NVIDIA, for generously supporting us with two GTX 1080 for developing a CUDA backend operator. 

- To JetBrains, for granting an open source license, allowing us to improve code quality with Resharper.

*We thank you.*

## Used libraries

For reference, a list of all libraries used in the Sigma project. The following libraries / frameworks are used in the core:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [log4net](https://logging.apache.org/log4net/) | Logging (log4j for .NET) |
| [NUnit](https://www.nunit.org/) | Unit testing |
| [DiffSharp](https://github.com/DiffSharp/DiffSharp), [Sigma.DiffSharp](https://github.com/GreekDictionary/Sigma.DiffSharp) | Functional automatic differentiation with ndarrays and various backends |
| [SharpZipLib](http://www.icsharpcode.net/) | Compression and decompression of various formats (zip, tar, gz, bzip, gzip) |
| [ManagedCuda](https://github.com/kunzmi/managedCuda), [ManagedCuda-CUBLAS](https://github.com/kunzmi/managedCuda) | Managed CUDA (GPU) and CuBLAS support |


The following libraries are used in the graphical and interactive visualiser:

| Library                             | Purpose                           |
| :-----------------------------------|:----------------------------------|
| [Dragablz](https://github.com/ButchersBoy/Dragablz) | Tearable tab control for WPF, which includes docking, tool windows |
| [LiveCharts](https://github.com/beto-rodriguez/Live-Charts), [LiveCharts.Wpf](https://github.com/beto-rodriguez/Live-Charts) | Charting, graphing, advanced data, plotting |
| [MahApps.Metro](https://github.com/MahApps/MahApps.Metro), [MahApps.Metro.Resources](https://github.com/MahApps/MahApps.Metro) | Toolkit for creating metro-style WPF applications |
| [MaterialDesignColors](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit), [MaterialDesignThemes.MahApps](https://github.com/ButchersBoy/MaterialDesignInXamlToolkit) | Material design for WPF/MahApps |
