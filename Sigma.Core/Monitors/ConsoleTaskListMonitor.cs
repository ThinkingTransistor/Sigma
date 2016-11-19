
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors
{
	/// <summary>
	/// A primitive task list monitor that writes the currently running tasks to the bottom of the console window.
	/// </summary>
	public class ConsoleTaskListMonitor : MonitorAdapter
	{
		public override void Start()
		{
			Thread consoleThread = new Thread(() =>
				{
					IList<string> printedTasks = new List<string>();

					while (true)
					{
						int oldCursorTop = Console.CursorTop;
						int oldCursorLeft = Console.CursorLeft;

						ConsoleColor oldBackgroundConsoleColor = Console.BackgroundColor;
						ConsoleColor oldForegroundConsoleColor = Console.ForegroundColor;
						bool oldCursorVisible = Console.CursorVisible;

						Console.SetCursorPosition(0, Console.WindowTop + Console.WindowHeight - 1);
						Console.CursorVisible = false;

						printedTasks.Clear();

						ICollection<ITaskObserver> runningTasks = SigmaEnvironment.TaskManager.GetTasks();

						foreach (ITaskObserver task in runningTasks)
						{
							string printedTask = task.Type.ExpressedType;

							if (task.Progress >= 0)
							{
								printedTask += $" ({task.Progress*100}%)";
							}

							printedTasks.Add(printedTask);
						}

						Console.Write(runningTasks.Count == 0 ? "No task is currently running." : string.Join(", ", printedTasks).PadRight(Console.WindowWidth, ' '));

						Console.SetCursorPosition(oldCursorLeft, oldCursorTop);

						Console.BackgroundColor = oldBackgroundConsoleColor;
						Console.ForegroundColor = oldForegroundConsoleColor;
						Console.CursorVisible = oldCursorVisible;

						Thread.Sleep(50);
					}
				}
			);

			consoleThread.Start();
		}
	}
}
