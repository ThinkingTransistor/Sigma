using System;
using System.Diagnostics;
using System.Threading;
using MahApps.Metro;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;
using System.Windows.Controls;
using MaterialDesignColors;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Color = System.Windows.Media.Color;
using Sigma.Core.Monitors.WPF.Control.Themes;

namespace Sigma.Tests.Internals.WPF
{
	class Program
	{
		static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));
			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.Tabs.AddTabs("Overview", "Data", "Tests");


			//sigma.Prepare()
			guiMonitor.Start();

			guiMonitor.Window.Dispatcher.Invoke(() => guiMonitor.Window.TitleCharacterCasing = CharacterCasing.Normal);


			var swatches = new SwatchesProvider().Swatches;

			StringBuilder builder = new StringBuilder();


			Random rand = new Random();

			while (true)
			{
				foreach (Swatch swatch in swatches)
				{
					Thread.Sleep(2000);

					guiMonitor.ColorManager.Dark = rand.Next(2) == 1;
					guiMonitor.ColorManager.PrimaryColor = swatch;

					string dark = guiMonitor.ColorManager.Dark ? "dark" : "light";

					Console.WriteLine($"Changing to: {dark} and {guiMonitor.ColorManager.PrimaryColor.Name}");
				}
			}


			//var swatches = new SwatchesProvider().Swatches;

			//StringBuilder builder = new StringBuilder();

			//foreach (Swatch swatch in swatches)
			//{
			//	builder.Append("public static readonly Swatch " + swatch.Name.ToUpper() + $"=new Swatch (\"{swatch.Name}\", new Hue[] " + "{ ");

			//	foreach (Hue hue in swatch.PrimaryHues)
			//	{
			//		builder.Append($"new Hue ({hue.Name}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " + "}), ");
			//	}

			//	builder.Append("}, new Hue[] { ");

			//	foreach (Hue hue in swatch.AccentHues)
			//	{
			//		builder.Append($"new Hue ({hue.Name}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Color.R}, G = {hue.Color.G}, B = {hue.Color.B} " + "}, " + "new Color() { " + $"A = {hue.Color.A}, R = {hue.Foreground.R}, G = {hue.Foreground.G}, B = {hue.Foreground.B} " + "}), ");
			//	}

			//	builder.Append("});\n");
			//}

			//File.WriteAllText(@"C:\Users\Plainer\Desktop\colors.txt", builder.ToString());

		}
	}
}
