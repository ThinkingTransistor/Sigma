using System;
using System.IO;
using System.Text;
using System.Threading;
using System.Windows.Controls;
using MaterialDesignColors;
using MaterialDesignThemes.Wpf;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;
using Sigma.Core.Monitors.WPF.Control.Themes;
using Sigma.Core.Monitors.WPF.View.Panels;
using Sigma.Core.Monitors.WPF.View.Tabs;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
		private class TestData
		{
			public string Name { get; set; }
			public int Epoch { get; set; }
		}

		private static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));
			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.AddTabs("Overview", "Data", "Tests");

			guiMonitor.WindowDispatcher((window) =>
			{
				window.TitleCharacterCasing = CharacterCasing.Normal;
			});

			sigma.Prepare();

			guiMonitor.WindowDispatcher((window) =>
			{


				TabUI tab = window.TabControl["Overview"];

				tab.AddCumulativeElement(new LineChartPanel("Control"), 2, 3);

				SimpleDataGridPanel<TestData> panel = new SimpleDataGridPanel<TestData>("Data");


				panel.Items.Add(new TestData { Name = "SomeOptimizer", Epoch = 14 });
				panel.Items.Add(new TestData { Name = "OtherOptimizer", Epoch = 1337 });

				tab.AddCumulativeElement(panel);


				//DataGridPanel panel2 = new DataGridPanel("compleX");
				//List<object> items = panel2.AddImageColumn("Images");
				//items.Add(new Image { Source = new BitmapImage(new Uri(@"C:\Users\Plainer\Desktop\sigma.png")) });


				CreateDefaultCards(window.TabControl["Tests"]);
			});

			guiMonitor.ColorManager.Dark = true;
			guiMonitor.ColorManager.PrimaryColor = MaterialDesignValues.Teal;
			guiMonitor.ColorManager.SecondaryColor = MaterialDesignValues.Amber;

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
				builder.Append("public static readonly Swatch " + swatch.Name.ToUpper() + $"=new Swatch (\"{swatch.Name}\", new [] " + "{ ");

				foreach (Hue hue in swatch.PrimaryHues)
				{
					builder.Append($"new Hue ({hue.Name}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " + "}), ");
				}

				builder.Append("}, new [] { ");

				foreach (Hue hue in swatch.AccentHues)
				{
					builder.Append($"new Hue ({hue.Name}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " + "}), ");
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

					guiMonitor.ColorManager.Dark = rand.Next(2) == 1;
					guiMonitor.ColorManager.Alternate = rand.Next(2) == 1;
					guiMonitor.ColorManager.PrimaryColor = swatch;

					string dark = guiMonitor.ColorManager.Dark ? "dark" : "light";

					Console.WriteLine($"Changing to: {dark} and {guiMonitor.ColorManager.PrimaryColor.Name}");
				}
			}
		}
	}
}
