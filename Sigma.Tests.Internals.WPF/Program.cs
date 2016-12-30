using System;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using log4net.Config;
using MaterialDesignColors;
using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Monitors;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Panels.Charts;
using Sigma.Core.Monitors.WPF.Panels.DataGrids;
using Sigma.Core.Monitors.WPF.Panels.Logging;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
		public static MinibatchIterator TrainingIterator;

		private static void Main(string[] args)
		{
			XmlConfigurator.Configure();

			SigmaEnvironment.Globals["web_proxy"] = WebUtils.GetProxyFromFileOrDefault(".customproxy");

			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));

			InitializeDownload(guiMonitor, sigma);

			IRegistry reg = new Registry(guiMonitor.Registry);
			guiMonitor.Registry.Add(StatusBarFactory.RegistryIdentifier, reg);
			reg.Add(StatusBarFactory.CustomFactoryIdentifier,
				new LambdaUIFactory(
					(app, window, param) =>
						new Label
						{
							Content = "Sigma is life, Sigma is love",
							Foreground = Brushes.White,
							VerticalAlignment = VerticalAlignment.Center,
							HorizontalAlignment = HorizontalAlignment.Center,
							FontSize = UIResources.P1
						}));

			guiMonitor.AddLegend(new StatusBarLegendInfo("Net test 1", MaterialColour.Red));
			StatusBarLegendInfo blueLegend = guiMonitor.AddLegend(new StatusBarLegendInfo("Netzzz", MaterialColour.Blue));
			guiMonitor.AddLegend(new StatusBarLegendInfo("Third net", MaterialColour.Green));

			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.AddTabs("Overview", "Data", "Tests", "Log2");

			guiMonitor.WindowDispatcher(window => { window.TitleCharacterCasing = CharacterCasing.Normal; });

			sigma.Prepare();


			guiMonitor.WindowDispatcher(window =>
			{
				window.TabControl["Data"].AddCumulativePanel(new LogTextPanel("Log"), 3, 4);
				window.TabControl["Log2"].AddCumulativePanel(new LogDataGrid("Log"), 3, 4);

				TabUI tab = window.TabControl["Overview"];

				tab.AddCumulativePanel(new LineChartPanel("Control"), 2, 3, guiMonitor.GetLegendInfo("Third net"));

				SimpleDataGridPanel<TestData> panel = new SimpleDataGridPanel<TestData>("Data");

				panel.Items.Add(new TestData { Name = "SomeOptimizer", Epoch = 14 });
				panel.Items.Add(new TestData { Name = "OtherOptimizer", Epoch = 1337 });

				tab.AddCumulativePanel(panel, legend: blueLegend);


				CustomDataGridPanel panel2 = new CustomDataGridPanel("compleX", "Picture", typeof(Image),
					nameof(ComplexTestData.Picture), "Text1", typeof(string), nameof(ComplexTestData.SomeText), "Text2", typeof(string),
					nameof(ComplexTestData.SomeOtherText), "Number", typeof(string), nameof(ComplexTestData.SomeInt));
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

				tab.AddCumulativePanel(new EmptyPanel("Empty panel"), 2);
			});


			//guiMonitor.ColourManager.Dark = true;
			//guiMonitor.ColourManager.Alternate = true;
			//guiMonitor.ColourManager.PrimaryColor = MaterialDesignValues.Blue;
			//guiMonitor.ColourManager.SecondaryColor = MaterialDesignValues.DeepPurple;

			//SwitchColor(monitor);
		}

		private static void InitializeDownload(IMonitor monitor, SigmaEnvironment sigma)
		{
			monitor.Registry["environment"] = sigma;

			ByteRecordReader mnistImageReader = new ByteRecordReader(headerLengthBytes: 16, recordSizeBytes: 28 * 28,
				source:
				new CompressedSource(new MultiSource(new FileSource("train-images-idx3-ubyte.gz"),
					new UrlSource("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))));

			ByteRecordExtractor mnistImageExtractor = mnistImageReader.Extractor("inputs", new[] { 0L, 0L }, new[] { 28L, 28L });

			ByteRecordReader mnistTargetReader = new ByteRecordReader(headerLengthBytes: 8, recordSizeBytes: 1,
				source:
				new CompressedSource(new MultiSource(new FileSource("train-labels-idx1-ubyte.gz"),
					new UrlSource("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))));
			IRecordExtractor mnistTargetExtractor =
				mnistTargetReader.Extractor("targets", new[] { 0L }, new[] { 1L })
					.Preprocess(new OneHotPreprocessor(0, 9));

			monitor.Registry["handler"] = new CpuFloat32Handler();

			Dataset dataset = new Dataset("mnist-training", Dataset.BlockSizeAuto, mnistImageExtractor, mnistTargetExtractor);
			IDataset[] slices = dataset.SplitRecordwise(0.8, 0.2);
			IDataset trainingData = slices[0];

			monitor.Registry["iterator"] = new MinibatchIterator(1, trainingData);
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

		private static void SwitchColor(WPFMonitor guiMonitor)
		{
			Random rand = new Random();

			while (true)
			{
				foreach (Swatch swatch in new SwatchesProvider().Swatches)
				{
					Thread.Sleep(4000);

					guiMonitor.ColourManager.Dark = rand.Next(2) == 1;
					guiMonitor.ColourManager.Alternate = rand.Next(2) == 1;
					guiMonitor.ColourManager.PrimaryColor = swatch;

					string dark = guiMonitor.ColourManager.Dark ? "dark" : "light";

					Console.WriteLine($@"Changing to: {dark} and {guiMonitor.ColourManager.PrimaryColor.Name}");
				}
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
	}
}