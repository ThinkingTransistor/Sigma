using System;
using System.Collections.Generic;
using System.Threading;
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
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Panels.Charts;
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
			Console.WriteLine(Thread.CurrentThread.CurrentUICulture);
			SigmaEnvironment.EnableLogging();
			SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");
			SigmaEnvironment sigma = SigmaEnvironment.Create("Sigma");

			WPFMonitor gui = sigma.AddMonitor(new WPFMonitor("WPF Monitor Demo"));
			gui.AddLegend(new StatusBarLegendInfo("Net", MaterialColour.LightBlue));
			gui.AddTabs("Overview", "Log");

			LineChartPanel lineChart = null;

			gui.WindowDispatcher(window =>
			{
				window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control"), 2, 1);

				lineChart = new LineChartPanel("Example");
				window.TabControl["Overview"].AddCumulativePanel(lineChart, 2, 2, gui.GetLegendInfo("Net"));

				window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log"), 3, 4);
			});

			sigma.Prepare();

			//IDataSource dataSource = new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"), new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")));

			//ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28, source: dataSource);
			//IRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L }).Preprocess(new NormalisingPreprocessor(0, 255));

			//IDataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor);
			//IDataset[] slices = dataset.SplitRecordwise(0.8, 0.2);
			//IDataset trainingData = slices[0];

			//IDataIterator iterator = new MinibatchIterator(10, trainingData);
			//foreach (var block in iterator.Yield(new CpuFloat32Handler(), sigma))
			//{
			//	//PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);
			//}

			//Console.WriteLine("+=+ Iterating over dataset again +=+");
			//Thread.Sleep(3000);

			//foreach (var block in iterator.Yield(new CpuFloat32Handler(), sigma))
			//{
			//	PrintFormattedBlock(block, PrintUtils.AsciiGreyscalePalette);
			//}

			Random rand = new Random();
			for (int i = 0; i < 30; i++)
			{
				Thread.Sleep(1000);

				gui.WindowDispatcher(window => lineChart.Add(rand.Next(7) + 3));
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