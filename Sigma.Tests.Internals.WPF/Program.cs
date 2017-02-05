using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading;
using System.Windows.Media;
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
using Sigma.Core.Monitors.WPF.Panels.DataGrids;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Monitors.WPF.View.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;
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

			gui.AddTabs("Overview", "Log", "Tests");

			gui.WindowDispatcher(window =>
			{
				TabUI tab = window.TabControl["Overview"];
			});

			LineChartPanel lineChart = null;
			CartesianChartPanel cartesianChart = null;

			//gui.AddLegend(new StatusBarLegendInfo("Net", MaterialColour.LightBlue));
			//gui.WindowDispatcher(window =>
			//{
			//	window.IsInitializing = true;
			//	window.TabControl["Overview"].AddCumulativePanel(new ControlPanel("Control"), 2, 1);

			//	//lineChart = new LineChartPanel("Example");
			//	//window.TabControl["Overview"].AddCumulativePanel(lineChart, 2, 2, gui.GetLegendInfo("Net"));
			//	cartesianChart = new CartesianChartPanel("Cartesian");
			//	window.TabControl["Overview"].AddCumulativePanel(cartesianChart, 2, 2, gui.GetLegendInfo("Net"));


			//	window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log"), 3, 4);
			//	window.IsInitializing = false;
			//});


			AddComplex(gui);

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

			//Random rand = new Random();
			//for (int i = 0; i < 1000; i++)
			//{
			//	Thread.Sleep(1000);
			//	try
			//	{
			//		gui.WindowDispatcher(window => cartesianChart.ChartValues.Add(rand.NextDouble() * 10));
			//	}
			//	catch (Exception)
			//	{
			//		return;
			//	}
			//}
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

		private class TestData
		{
			public string Name { get; set; }
			public int Epoch { get; set; }
		}

		private class ComplexTestData
		{
			public ImageSource Picture { get; set; }
			public string SomeText { get; set; }
			public string SomeOtherText { get; set; }
			public int SomeInt { get; set; }
		}


		private static void AddComplex(WPFMonitor guiMonitor)
		{
			StatusBarLegendInfo blueLegend = guiMonitor.AddLegend(new StatusBarLegendInfo("Net", MaterialColour.Blue));
			guiMonitor.AddLegend(new StatusBarLegendInfo("Third net", MaterialColour.Red));
			guiMonitor.AddLegend(new StatusBarLegendInfo("Net test 1", MaterialColour.Pink));

			guiMonitor.WindowDispatcher(window =>
			{
				window.IsInitializing = true;
				window.TabControl["Log"].AddCumulativePanel(new LogDataGridPanel("Log"), 3, 4);
				TabUI tab = window.TabControl["Overview"];

				tab.AddCumulativePanel(new LineChartPanel("Control"), 2, 3, guiMonitor.GetLegendInfo("Third net"));

				SimpleDataGridPanel<TestData> panel = new SimpleDataGridPanel<TestData>("Data");

				panel.Items.Add(new TestData { Name = "SomeOptimizer", Epoch = 14 });
				panel.Items.Add(new TestData { Name = "OtherOptimizer", Epoch = 1337 });

				tab.AddCumulativePanel(panel, legend: blueLegend);


				CustomDataGridPanel panel2 = new CustomDataGridPanel("compleX", "Picture", typeof(Image), nameof(ComplexTestData.Picture), "Text1", typeof(string), nameof(ComplexTestData.SomeText), "Text2", typeof(string), nameof(ComplexTestData.SomeOtherText), "Number", typeof(string), nameof(ComplexTestData.SomeInt));
				ComplexTestData data = new ComplexTestData
				{
					//Picture = new BitmapImage(new Uri(@"C:\Users\Flo\Dropbox\Diplomarbeit\Logo\export\128x128.png")),
					SomeInt = 12,
					SomeOtherText = "other",
					SomeText = "text"
				};

				panel2.Content.Items.Add(data);

				tab.AddCumulativePanel(panel2, 1, 3, guiMonitor.GetLegendInfo("Net test 1"));

				CreateDefaultCards(window.TabControl["Tests"]);

				tab.AddCumulativePanel(new ControlPanel("Control panel"), 2);
				window.IsInitializing = false;
			});
		}

		private static void CreateDefaultCards(TabUI tab)
		{
			for (int i = 0; i < 3; i++)
			{
				tab.AddCumulativePanel(new EmptyPanel($"Card No. {i}"));
			}

			tab.AddCumulativePanel(new EmptyPanel("Big card"), 3);
			tab.AddCumulativePanel(new EmptyPanel("Tall card"), 2);
			for (int i = 0; i < 2; i++)
			{
				tab.AddCumulativePanel(new EmptyPanel("Small card"));
			}

			tab.AddCumulativePanel(new EmptyPanel("Wide card"), 1, 2);
		}
	}
}