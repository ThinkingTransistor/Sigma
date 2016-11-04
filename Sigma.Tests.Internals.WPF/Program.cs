
using System.Threading;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;
using System.Windows.Controls;
using System.Windows.Media;
using System.Diagnostics;
using MaterialDesignColors;
using System.Text;
using System;
using Sigma.Core.Monitors.WPF.Control.Themes;
using System.Linq;
namespace Sigma.Tests.Internals.WPF
{
	class Program
	{
		static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));
			guiMonitor.Priority = ThreadPriority.Highest;
			guiMonitor.AddTabs("Overview", "Data", "Tests");


			//sigma.Prepare()
			guiMonitor.Start();


			guiMonitor.WindowDispatcher((window) =>
			{
				window.TitleCharacterCasing = CharacterCasing.Normal;


				//window.GlowBrush = window.FindResource("AccentColorBrush") as Brush;
			});

			guiMonitor.ColorManager.PrimaryColor = MaterialDesignSwatches.BLUE;




			Random rand = new Random();

			while (true)
			{
				foreach (Swatch swatch in new SwatchesProvider().Swatches)
				{
					Thread.Sleep(1500);

					//guiMonitor.ColorManager.Dark = rand.Next(2) == 1;
					guiMonitor.ColorManager.PrimaryColor = swatch;

					string dark = guiMonitor.ColorManager.Dark ? "dark" : "light";

					Console.WriteLine($"Changing to: {dark} and {guiMonitor.ColorManager.PrimaryColor.Name}");
				}
			}


			//StringBuilder builder = new StringBuilder();
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
