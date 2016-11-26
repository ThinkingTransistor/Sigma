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
using Sigma.Core.Monitors.WPF.View.Tabs;

namespace Sigma.Tests.Internals.WPF
{
	public class Program
	{
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

				tab.AddCumulativeElement(new TestPanel("Control") { Content = "TestLabel" });
				//CreateDefaultCards(tab);
			});

			guiMonitor.ColorManager.Dark = true;
			guiMonitor.ColorManager.PrimaryColor = MaterialDesignSwatches.Teal;
			guiMonitor.ColorManager.SecondaryColor = MaterialDesignSwatches.Amber;

			while (true)
			{
				Thread.Sleep(1500);

				guiMonitor.ColorManager.Dark = !guiMonitor.ColorManager.Dark;
			}

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
