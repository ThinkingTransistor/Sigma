using System.Collections.Generic;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.ViewModel.StatusBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	public class TaskVisualizerFactory : IUIFactory<UIElement>
	{
		private readonly int _maxElements;

		private readonly List<IWpfTaskVisualizationManager> _visualizationManagers;

		private readonly IUIFactory<UIElement> _showMoreFactory = new ShowMoreFactory();

		public TaskVisualizerFactory(int maxElements, IUIFactory<UIElement> showMoreFactory = null)
		{
			_maxElements = maxElements;
			_visualizationManagers = new List<IWpfTaskVisualizationManager>();

			if (showMoreFactory != null)
			{
				_showMoreFactory = showMoreFactory;
			}
		}

		public UIElement CreatElement(Application app, Window window, params object[] parameters)
		{
			IWpfTaskVisualizationManager manager;
			UIElement showMoreIndicator = _showMoreFactory.CreatElement(app, window, parameters);

			lock (_visualizationManagers)
			{
				IEnumerable<ITaskObserver> activeTasks = _visualizationManagers.Count > 0 ? _visualizationManagers[0].ActiveTasks : null;
				IEnumerable<ITaskObserver> pendingTasks = _visualizationManagers.Count > 0 ? _visualizationManagers[0].PendingTasks : null;

				manager = new TaskVisualizationManager(_maxElements, showMoreIndicator, SigmaEnvironment.TaskManager, activeTasks, pendingTasks);

				_visualizationManagers.Add(manager);
			}

			// dispose the manager on close
			window.Closed += (sender, args) =>
			{
				manager.Dispose();

				lock (_visualizationManagers)
				{
					_visualizationManagers.Remove(manager);
				}
			};

			Grid grid = new Grid();

			// + 1 for the "..." label
			for (int i = 0; i <= _maxElements; i++)
			{
				grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
			}

			for (int i = 0; i < _maxElements; i++)
			{
				grid.Children.Add(manager.TaskVisualizers[i]);
				Grid.SetColumn(manager.TaskVisualizers[i], i);
			}

			grid.Children.Add(showMoreIndicator);
			Grid.SetColumn(showMoreIndicator, _maxElements);

			return grid;
		}

		private class ShowMoreFactory : IUIFactory<UIElement>
		{
			public UIElement CreatElement(Application app, Window window, params object[] parameters)
			{
				return new Label { Content = "more" };
			}
		}
	}
}