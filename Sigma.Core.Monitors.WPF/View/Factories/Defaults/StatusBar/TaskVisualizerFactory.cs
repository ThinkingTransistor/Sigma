using System.Windows;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	public class TaskVisualizerFactory : IUIFactory<TaskVisualizer>
	{
		public TaskVisualizer CreatElement(Application app, Window window, params object[] parameters)
		{
			return new TaskVisualizer();
		}
	}
}