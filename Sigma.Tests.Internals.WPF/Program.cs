using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows.Controls;
using System.Windows.Media;
using MaterialDesignColors;
using MaterialDesignThemes.Wpf;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Control.Factories;
using Sigma.Core.Monitors.WPF.Control.Themes;
using Sigma.Core.Monitors.WPF.Control.TitleBar;
using Sigma.Core.Monitors.WPF.View.Panels;
using Sigma.Core.Monitors.WPF.View.Panels.DataGrids;
using Sigma.Core.Monitors.WPF.View.Tabs;
using Sigma.Core.Monitors.WPF.View.Windows;

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

			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.AddTabs("Overview", "Data", "Tests");

			guiMonitor.WindowDispatcher(window => { window.TitleCharacterCasing = CharacterCasing.Normal; });

			sigma.Prepare();

			guiMonitor.WindowDispatcher((window) =>
			{
				TabUI tab = window.TabControl["Overview"];

				tab.AddCumulativeElement(new LineChartPanel("Control"), 2, 3);

				SimpleDataGridPanel<TestData> panel = new SimpleDataGridPanel<TestData>("Data");

				panel.Items.Add(new TestData { Name = "SomeOptimizer", Epoch = 14 });
				panel.Items.Add(new TestData { Name = "OtherOptimizer", Epoch = 1337 });

				tab.AddCumulativeElement(panel);


				CustomDataGridPanel panel2 = new CustomDataGridPanel("compleX", "Picture", typeof(Image), nameof(ComplexTestData.Picture), "Text1", typeof(string), nameof(ComplexTestData.SomeText), "Text2", typeof(string), nameof(ComplexTestData.SomeOtherText), "Number", typeof(string), nameof(ComplexTestData.SomeInt));
				ComplexTestData data = new ComplexTestData
				{
					//Picture = new BitmapImage(new Uri(@"C:\Users\Flo\Dropbox\Diplomarbeit\Logo\export\128x128.png")),
					SomeInt = 12,
					SomeOtherText = "other",
					SomeText = "text"
				};

				panel2.Content.Items.Add(data);

				//panel2.AddImageColumn("Images", nameof(ComplexTestData.Picture));
				//panel2.Content.Items.Add(
				//	new ComplexTestData { SomeText = "tesstkd", Picture = new BitmapImage(new Uri(@"C:\Users\Plain\Desktop\sigma2.png")) });

				//items.Add(new Image {Source = new BitmapImage(new Uri(@"C:\Users\Plain\Desktop\sigma2.png"))}.Source);
				//items.Add(new Image { Source = new BitmapImage(new Uri(@"C:\Users\Plain\Desktop\sigma.png")) });
				tab.AddCumulativeElement(panel2, 1, 3);

				CreateDefaultCards(window.TabControl["Tests"]);
			});

			guiMonitor.ColourManager.Dark = true;
			guiMonitor.ColourManager.PrimaryColor = MaterialDesignValues.Teal;
			guiMonitor.ColourManager.SecondaryColor = MaterialDesignValues.Amber;

			//SwitchColor(guiMonitor);

			//ExtractSwatches();
		}

		private static void CreateDefaultCards(TabUI tab)
		{
			for (int i = 0; i < 3; i++)
			{
				tab.AddCumulativeElement(new Card { Content = "Card No. " + i });
			}

			tab.AddCumulativeElement(new Card { Content = "Big card" }, 3);
			tab.AddCumulativeElement(new Card { Content = "Tall card" }, 2);
			for (int i = 0; i < 2; i++)
			{
				tab.AddCumulativeElement(new Card { Content = "Small card" });
			}

			tab.AddCumulativeElement(new Card { Content = "Wide card" }, 1, 2);
		}

		private static void ExtractSwatches()
		{
			StringBuilder builder = new StringBuilder();
			var swatches = new SwatchesProvider().Swatches;

			foreach (Swatch swatch in swatches)
			{
				builder.Append("public static readonly Swatch " + swatch.Name.ToUpper() + $"=new Swatch (\"{swatch.Name}\", new [] " +
							   "{ ");

				foreach (Hue hue in swatch.PrimaryHues)
				{
					builder.Append($"new Hue ({hue.Name}, " + "new Color() { " +
								   $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " +
								   "new Color() { " +
								   $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " +
								   "}), ");
				}

				builder.Append("}, new [] { ");

				foreach (Hue hue in swatch.AccentHues)
				{
					builder.Append($"new Hue ({hue.Name}, " + "new Color() { " +
								   $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " +
								   "new Color() { " +
								   $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " +
								   "}), ");
				}

				builder.Append("});\n");
			}

			File.WriteAllText(@"C:\Users\Plainer\Desktop\colors.txt", builder.ToString());
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

					Console.WriteLine($"Changing to: {dark} and {guiMonitor.ColourManager.PrimaryColor.Name}");
				}
			}
		}
	}
}
