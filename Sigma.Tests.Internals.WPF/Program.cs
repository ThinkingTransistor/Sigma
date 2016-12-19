using System;
using System.Diagnostics;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using MaterialDesignColors;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Model.UI.Resources;
using Sigma.Core.Monitors.WPF.Model.UI.StatusBar;
using Sigma.Core.Monitors.WPF.Panels;
using Sigma.Core.Monitors.WPF.Panels.DataGrids;
using Sigma.Core.Monitors.WPF.View.Factories;
using Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar;
using Sigma.Core.Monitors.WPF.ViewModel.Tabs;
using Sigma.Core.Utils;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
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

		private static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));

			IRegistry reg = new Registry(guiMonitor.Registry);
			guiMonitor.Registry.Add(StatusBarFactory.RegistryIdentifier, reg);
			reg.Add(StatusBarFactory.CustomFactoryIdentifier, new LambdaUIFactory((app, window, param) => new Label { Content = "Sigma is life, Sigma is love", Foreground = Brushes.White, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center, FontSize = UIResources.P1 }));

			Debug.WriteLine(guiMonitor.Registry);

			guiMonitor.AddLegend(new StatusBarLegendInfo("Net test 1", MaterialColour.Red));
			StatusBarLegendInfo blueLegend = guiMonitor.AddLegend(new StatusBarLegendInfo("Netzzz", MaterialColour.Blue));
			guiMonitor.AddLegend(new StatusBarLegendInfo("Third net", MaterialColour.Green));

			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.AddTabs("Overview", "Data", "Tests");

			guiMonitor.WindowDispatcher(window => { window.TitleCharacterCasing = CharacterCasing.Normal; });

			sigma.Prepare();

			guiMonitor.WindowDispatcher(window =>
			{
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

				tab.AddCumulativePanel(new EmptyPanel("Empty panel"), 2);
			});

			guiMonitor.ColourManager.Dark = true;
			guiMonitor.ColourManager.Alternate = true;
			guiMonitor.ColourManager.PrimaryColor = MaterialDesignValues.BlueGrey;
			guiMonitor.ColourManager.SecondaryColor = MaterialDesignValues.Amber;

			//SwitchColor(guiMonitor);
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
	}
}
