using Sigma.Core.Monitors.WPF;

namespace Sigma.Tests.Internals.WPF
{
	class Program
	{
		static void Main(string[] args)
		{
			WPFMonitor guiMonitor = new WPFMonitor("Sigma GUI Demo");
			guiMonitor.Priority = System.Threading.ThreadPriority.Highest;


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
