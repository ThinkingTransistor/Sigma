using System;
using System.Collections.Generic;
using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.MathAbstract;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Panels.Control;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
		public static MinibatchIterator TrainingIterator;

		private static void Main(string[] args)
		{
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("WPF Monitor Demo"));
			gui.AddTabs("Overview", "Log");

			sigma.Prepare();

			gui.WindowDispatcher(window =>
			{
				window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control"), 2, 1);
				window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log"), 3, 4);
			});

			IDataSource dataSource = new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")));

			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: dataSource);
			IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			IDataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor);
			IDataset[] slices = dataset.SplitRecordwise(0.8, 0.2);
			IDataset trainingData = slices[0];

			IDataIterator iterator = new MinibatchIterator(10, trainingData);
			foreach (var block in iterator.Yield(new CpuFloat32Handler(), sigma))
			{
				PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);
			}
		}

		private static void PrintFormattedBlock(IDictionary<string, INDArray> block, char[] palette)
		{
			foreach (string name in block.Keys)
			{
				string blockString = name == "inputs"
					? ArrayUtils.ToString<float>(block[name], e => palette[(int) (e * (palette.Length - 1))].ToString(), maxDimensionNewLine: 0, printSeperator: false)
					: block[name].ToString();

				Console.WriteLine($"[{name}]=\n" + blockString);
			}
		} 
	}
}