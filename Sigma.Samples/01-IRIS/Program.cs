
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
using Sigma.Core.Training;
using Sigma.Core.Training.Hooks.Reporters;
using Sigma.Core.Training.Initialisers;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Optimisers;
using Sigma.Core.Training.Optimisers.Gradient;
using Sigma.Core.Utils;

namespace _01_IRIS
{
	internal class Program
	{
		private static void Main()
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma-IRIS");


			ITrainer trainer = CreateIrisTrainer(sigma);
		}

		/// <summary>
		/// Create an IRIS trainer that observers the current epoch and iteration
		/// </summary>
		/// <param name="sigma">The sigma environemnt.</param>
		/// <returns>The newly created trainer that can be added to the environemnt.</returns>
		private static ITrainer CreateIrisTrainer(SigmaEnvironment sigma)
		{
			CsvRecordReader irisReader = new CsvRecordReader(new MultiSource(new FileSource("iris.data"), new UrlSource("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")));
			IRecordExtractor irisExtractor = irisReader.Extractor("inputs", new[] { 0, 3 }, "targets", 4).AddValueMapping(4, "Iris-setosa", "Iris-versicolor", "Iris-virginica");
			irisExtractor = irisExtractor.Preprocess(new OneHotPreprocessor(sectionName: "targets", minValue: 0, maxValue: 2));
			irisExtractor = irisExtractor.Preprocess(new PerIndexNormalisingPreprocessor(0, 1, "inputs", 0, 4.3, 7.9, 1, 2.0, 4.4, 2, 1.0, 6.9, 3, 0.1, 2.5));

			Dataset dataset = new Dataset("iris", Dataset.BlockSizeAuto, irisExtractor);
			IDataset trainingDataset = dataset;
			IDataset validationDataset = dataset;

			ITrainer trainer = sigma.CreateTrainer("test");

			trainer.Network = new Network
			{
				Architecture = InputLayer.Construct(4)
								+ FullyConnectedLayer.Construct(10)
								+ FullyConnectedLayer.Construct(20)
								+ FullyConnectedLayer.Construct(10)
								+ FullyConnectedLayer.Construct(3)
								+ OutputLayer.Construct(3)
								+ SquaredDifferenceCostLayer.Construct()
			};
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
