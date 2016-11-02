using System;
using Sigma.Core;
using Sigma.Core.Monitors.WPF;

namespace Sigma.Tests.Internals.WPF
{
	class Program
	{
		static void Main(string[] args)
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			WPFMonitor guiMonitor = sigma.AddMonitor(new WPFMonitor("Sigma GUI Demo"));
			guiMonitor.Priority = System.Threading.ThreadPriority.Highest;
			guiMonitor.Tabs.AddTabs("Test", "Test");

			

			Console.WriteLine(guiMonitor.Tabs.ContainsTab("Test"));
			Console.WriteLine("Title of tab test: " + guiMonitor.Tabs["Test"].Title);

			//gui.DefaultGridSize = {3, 4};
			//gui.AddTabs("Overview", "Data", "Tests");
			//gui.PrimaryColor = Colors.GreyBlue;


			//sigma.Prepare()
			guiMonitor.Start();

			//Console.WriteLine("Test");
			//Console.ReadKey();
		}
	}
}
