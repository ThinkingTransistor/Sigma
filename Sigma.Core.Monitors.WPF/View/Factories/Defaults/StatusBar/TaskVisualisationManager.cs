using System.Collections.Generic;
using System.Diagnostics;
using System.Windows;
using System.Windows.Controls;
using Sigma.Core.Monitors.WPF.ViewModel.CustomControls.StatusBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.View.Factories.Defaults.StatusBar
{
	public class TaskVisualisationManager : IUIFactory<UIElement>
	{
		private readonly TaskVisualizer[] _taskVisualizers;

		private readonly List<ITaskObserver> _pendingTasks;

		private int _activeTasks;

		private ITaskManager _taskManager;

		private Label _moreLabel;

		public ITaskManager TaskManager
		{
			get { return _taskManager; }
			set
			{
				if (_taskManager != null)
				{
					TaskManager.TaskCreated -= TaskCreated;
					TaskManager.TaskEnded -= TaskStopped;
					TaskManager.TaskCanceled -= TaskStopped;
				}

				if (value != null)
				{
					value.TaskCreated += TaskCreated;
					value.TaskEnded += TaskStopped;
					value.TaskCanceled += TaskStopped;
				}

				_taskManager = value;
			}
		}

		public TaskVisualisationManager(int maxElements)
		{
			_taskVisualizers = new TaskVisualizer[maxElements];
			_pendingTasks = new List<ITaskObserver>();
			TaskManager = SigmaEnvironment.TaskManager;

			Debug.WriteLine("TaskVisualizer created");
		}

		public UIElement CreatElement(Application app, Window window, params object[] parameters)
		{
			StackPanel tasks = new StackPanel
			{
				Orientation = Orientation.Horizontal,
				HorizontalAlignment = HorizontalAlignment.Center,
				Margin = new Thickness(0, 0, 30, 0)
			};

			for (int i = 0; i < _taskVisualizers.Length; i++)
			{
				_taskVisualizers[i] = new TaskVisualizer();

				tasks.Children.Add(_taskVisualizers[i]);
			}

			_moreLabel = new Label { Content = "more", Visibility = Visibility.Hidden };

			tasks.Children.Add(_moreLabel);

			return tasks;
		}

		private void ShowMoreLabel(bool show)
		{
			_moreLabel.Dispatcher.Invoke(() =>
			{
				_moreLabel.Visibility = show ? Visibility.Visible : Visibility.Hidden;
			});
		}

		public void TaskCreated(object sender, TaskModifiedEventArgs args)
		{
			_pendingTasks.Add(args.Task);

			UpdateTasks();
		}

		public void TaskStopped(object sender, TaskModifiedEventArgs args)
		{
			if (RemoveTask(args.Task))
			{
				UpdateTasks();
			}
		}

		private ITaskObserver[] VisualizedTasks()
		{
			ITaskObserver[] observers = new ITaskObserver[_taskVisualizers.Length];

			for (int i = 0; i < _taskVisualizers.Length; i++)
			{
				observers[i] = _taskVisualizers[i].ActiveTask;
			}

			return observers;
		}

		private void UpdateTasks()
		{
			ITaskObserver[] visualizedTasks = VisualizedTasks();

			// fill null tasks
			for (int i = 0; i < visualizedTasks.Length; i++)
			{
				if (visualizedTasks[i] == null)
				{
					for (int j = i; j < visualizedTasks.Length - 1; j++)
					{
						visualizedTasks[j] = visualizedTasks[j + 1];
					}

					visualizedTasks[visualizedTasks.Length - 1] = null;
				}
			}

			// add pending tasks
			if (_pendingTasks.Count > 0)
			{
				for (int i = 0; i < visualizedTasks.Length; i++)
				{
					if (visualizedTasks[i] == null)
					{
						visualizedTasks[i] = _pendingTasks[0];

						_pendingTasks.RemoveAt(0);
						if (_pendingTasks.Count <= 0)
						{
							break;
						}
					}
				}
			}

			_activeTasks = 0;
			for (int i = 0; i < visualizedTasks.Length; i++)
			{
				if (visualizedTasks[i] == null)
				{
					break;
				}

				_activeTasks++;
			}

			ShowMoreLabel(_activeTasks == _taskVisualizers.Length && _pendingTasks.Count > 0);

			// set new active tasks
			for (int i = 0; i < _taskVisualizers.Length; i++)
			{
				_taskVisualizers[i].Dispatcher.Invoke(() => _taskVisualizers[i].SetActive(visualizedTasks[i]));
			}
		}

		private bool RemoveTask(ITaskObserver task)
		{
			if (_pendingTasks.Contains(task))
			{
				_pendingTasks.Remove(task);

				return true;
			}

			foreach (TaskVisualizer taskVisualizer in _taskVisualizers)
			{
				if (ReferenceEquals(taskVisualizer.ActiveTask, task))
				{
					taskVisualizer.Dispatcher.Invoke(() => taskVisualizer.SetActive(null));

					return true;
				}
			}

			return false;
		}

		~TaskVisualisationManager()
		{
			// remove the events
			TaskManager = null;

			// remove the references to all TaskVisualizers
			for (int i = 0; i < _taskVisualizers.Length; i++)
			{
				_taskVisualizers[i] = null;
			}

			// clear the pending tasks
			_pendingTasks.Clear();
		}
	}
}