/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using Sigma.Core.Monitors.WPF.View.CustomControls.StatusBar;
using Sigma.Core.Utils;

namespace Sigma.Core.Monitors.WPF.ViewModel.StatusBar
{
	/// <summary>
	///     This class manages multiple <see cref="TaskVisualizer" />s and listens to the
	///     events of a given <see cref="TaskManager" />. When an event occurs, a correct
	///     action is taken to set <see cref="TaskVisualizer" /> visible.
	/// </summary>
	public class TaskVisualizationManager : IWpfTaskVisualizationManager
	{
		/// <summary>
		///     A list of the pending tasks.
		/// </summary>
		private readonly List<ITaskObserver> _pendingTasks;

		/// <summary>
		///     The <see cref="TaskManager" />. It should be assigned with
		///     the public variable "<see cref="TaskManager" />" in order to
		///     listen to events.
		/// </summary>
		private ITaskManager _taskManager;

		/// <summary>
		///     Create a <see cref="TaskVisualizationManager" /> with a given amount of <see cref="maxElements" />.
		///     <see cref="SigmaEnvironment.TaskManager" /> is used as <see cref="TaskManager" />. No active or pending
		///     tasks are passed.
		/// </summary>
		/// <param name="maxElements">The amount of <see cref="maxElements" />.</param>
		/// <param name="moreTasksIndicator">The indicator that will be shown when too many tasks are visible. </param>
		public TaskVisualizationManager(int maxElements, UIElement moreTasksIndicator)
			: this(maxElements, moreTasksIndicator, SigmaEnvironment.TaskManager, null, null)
		{
		}

		/// <summary>
		///     Create a <see cref="TaskVisualizationManager" /> with a given amount of <see cref="maxElements" />.
		/// </summary>
		/// <param name="maxElements">The amount of <see cref="maxElements" />.</param>
		/// <param name="moreTasksIndicator">The indicator that will be shown when too many tasks are visible.</param>
		/// <param name="taskManager">The <see cref="TaskManager" /> to use.</param>
		/// <param name="activeTasks">The tasks that are currently active. Pass <c>null</c> if no tasks should be added.</param>
		/// <param name="pendingTasks">The tasks that are currently pending. </param>
		public TaskVisualizationManager(int maxElements, UIElement moreTasksIndicator, ITaskManager taskManager,
			IEnumerable<ITaskObserver> activeTasks, IEnumerable<ITaskObserver> pendingTasks)
		{
			if (maxElements <= 0)
			{
				throw new ArgumentOutOfRangeException(nameof(maxElements));
			}
			if (moreTasksIndicator == null)
			{
				throw new ArgumentNullException(nameof(moreTasksIndicator));
			}

			moreTasksIndicator.Visibility = Visibility.Hidden;

			ActiveTasks = new ITaskObserver[maxElements];
			_pendingTasks = new List<ITaskObserver>();

			TaskVisualizers = new TaskVisualizer[maxElements];

			for (int i = 0; i < TaskVisualizers.Length; i++)
			{
				TaskVisualizers[i] = new TaskVisualizer();
			}

			MoreTasksIndicator = moreTasksIndicator;

			if (activeTasks != null)
			{
				_pendingTasks.AddRange(activeTasks);
			}

			if (pendingTasks != null)
			{
				_pendingTasks.AddRange(pendingTasks);
			}

			UpdateTasks();

			// when everything is set up, begin listening to the events 
			TaskManager = taskManager;
		}

		/// <summary>
		///     The <see cref="TaskManager" />. When set it removes old event
		///     listens and adds events to the new one.
		/// </summary>
		public ITaskManager TaskManager
		{
			get { return _taskManager; }
			set
			{
				// if there was already a taskmanager
				if (_taskManager != null)
				{
					TaskManager.TaskCreated -= TaskCreated;
					TaskManager.TaskEnded -= TaskStopped;
					TaskManager.TaskCanceled -= TaskStopped;
				}

				// if there is a new taskmanager 
				if (value != null)
				{
					value.TaskCreated += TaskCreated;
					value.TaskEnded += TaskStopped;
					value.TaskCanceled += TaskStopped;
				}

				_taskManager = value;
			}
		}

		/// <summary>
		///     This variable tells how many tasks are currently visualized.
		///     This methods <em>counts</em> the active items, so cache it if you use
		///     it often.
		/// </summary>
		public int ActiveTasksCount
		{
			get { return ActiveTasks.Count(t => t != null); }
		}

		/// <summary>
		///     The <see cref="TaskVisualizer" />s to keep track of.
		/// </summary>
		public TaskVisualizer[] TaskVisualizers { get; }

		/// <summary>
		///     The indicator that will be shown when too many tasks are visible.
		/// </summary>
		public UIElement MoreTasksIndicator { get; }

		/// <summary>
		///     An array of the active <see cref="ITaskObserver" />s.
		///     If there is no task, for an index, the task is null.
		/// </summary>
		public ITaskObserver[] ActiveTasks { get; }

		/// <summary>
		///     A list of the pending tasks.
		/// </summary>
		List<ITaskObserver> IWpfTaskVisualizationManager.PendingTasks => _pendingTasks;

		public void Dispose()
		{
			// remove the events
			TaskManager = null;

			// remove the references to all TaskVisualizers
			for (int i = 0; i < TaskVisualizers.Length; i++)
			{
				TaskVisualizers[i] = null;
			}

			// clear the pending tasks
			_pendingTasks.Clear();
		}

		/// <summary>
		///     The function that will be called when a new task is created.
		/// </summary>
		/// <param name="sender">The raiser of the event.</param>
		/// <param name="args">The event arguments.</param>
		public void TaskCreated(object sender, TaskModifiedEventArgs args)
		{
			AddTask(args.Task);

			UpdateTasks();
		}

		/// <summary>
		///     The function that will be called when a task is stopped or
		///     cancelled.
		/// </summary>
		/// <param name="sender">The raiser of the event.</param>
		/// <param name="args">The event arguments.</param>
		public void TaskStopped(object sender, TaskModifiedEventArgs args)
		{
			if (RemoveTask(args.Task))
			{
				UpdateTasks();
			}
		}

		/// <summary>
		///     This methods updates the tasks and should be called after
		///     a new <see cref="ITaskObserver" /> has been added or removed.
		///     In this function, the tasks will be aligned in order,
		///     <see cref="_pendingTasks" /> will be added to empty spots,
		///     and the new <see cref="ITaskObserver" /> will be set for
		///     the <see cref="TaskVisualizers" />.
		/// </summary>
		private void UpdateTasks()
		{
			lock (ActiveTasks)
			{
				// fill null tasks
				for (int i = 0; i < ActiveTasks.Length; i++)
				{
					if (ActiveTasks[i] == null)
					{
						for (int j = i; j < ActiveTasks.Length - 1; j++)
						{
							ActiveTasks[j] = ActiveTasks[j + 1];
						}

						ActiveTasks[ActiveTasks.Length - 1] = null;
					}
				}

				// add pending tasks
				if (_pendingTasks.Count > 0)
				{
					for (int i = 0; i < ActiveTasks.Length; i++)
					{
						if (ActiveTasks[i] == null)
						{
							// add task to active tasks
							ActiveTasks[i] = _pendingTasks[0];

							_pendingTasks.RemoveAt(0);
							if (_pendingTasks.Count <= 0)
							{
								break;
							}
						}
					}
				}

				ShowMoreIndicator((ActiveTasksCount == TaskVisualizers.Length) && (_pendingTasks.Count > 0));

				// set new active tasks
				for (int i = 0; i < TaskVisualizers.Length; i++)
				{
					int currentTask = i;
					TaskVisualizers[i].Dispatcher.Invoke(() => TaskVisualizers[currentTask].SetActive(ActiveTasks[currentTask]));
				}
			}
		}

		private void ShowMoreIndicator(bool show)
		{
			MoreTasksIndicator.Dispatcher.Invoke(
				() => MoreTasksIndicator.Visibility = show ? Visibility.Visible : Visibility.Hidden);
		}

		/// <summary>
		///     This method adds a <see cref="ITaskObserver" />
		///     to the <see cref="_pendingTasks" />.
		/// </summary>
		/// <param name="task">The <see cref="ITaskObserver" /> that should be added.</param>
		private void AddTask(ITaskObserver task)
		{
			_pendingTasks.Add(task);
		}

		/// <summary>
		///     Remove a <see cref="ITaskObserver" />. This <see cref="ITaskObserver" /> can be
		///     in the <see cref="_pendingTasks" /> or in the <see cref="ActiveTasks" />.
		/// </summary>
		/// <param name="task">The <see cref="ITaskObserver" /> that should be removed. </param>
		/// <returns></returns>
		private bool RemoveTask(ITaskObserver task)
		{
			if (_pendingTasks.Contains(task))
			{
				_pendingTasks.Remove(task);

				return true;
			}

			for (int i = 0; i < TaskVisualizers.Length; i++)
			{
				if (ReferenceEquals(TaskVisualizers[i].ActiveTask, task))
				{
					TaskVisualizers[i].Dispatcher.Invoke(() => TaskVisualizers[i].SetActive(null));

					// remove tasks from active tasks
					ActiveTasks[i] = null;

					return true;
				}
			}

			return false;
		}

		~TaskVisualizationManager()
		{
			Dispose();
		}
	}
}