/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Core.Utils
{
	public class TaskModifiedEventArgs : EventArgs
	{
		public ITaskObserver Task { get; }
		public TaskObserveStatus NextStatus { get; }

		public TaskModifiedEventArgs(ITaskObserver task, TaskObserveStatus nextStatus)
		{
			Task = task;
			NextStatus = nextStatus;
		}
	}

	/// <summary>
	/// A default implementation of the <see cref="ITaskManager"/> interface. 
	/// Represents a task manager, that manages a set of task observers. 
	/// </summary>
	public sealed class TaskManager : ITaskManager
	{
		public event EventHandler<TaskModifiedEventArgs> TaskCreated;
		public event EventHandler<TaskModifiedEventArgs> TaskCanceled;
		public event EventHandler<TaskModifiedEventArgs> TaskEnded;

		private readonly IList<ITaskObserver> _runningObservers = new List<ITaskObserver>();

		public ITaskObserver BeginTask(ITaskType taskType, string taskDescription = null, bool exposed = true, bool indeterminate = true)
		{
			if (taskType == null)
			{
				throw new ArgumentNullException(nameof(taskType));
			}

			ITaskObserver observer = new TaskObserver(taskType, taskDescription, exposed);
			OnTaskCreated(observer);

			observer.Status = TaskObserveStatus.Running;
			observer.StartTime = DateTime.Now;

			if (indeterminate)
			{
				observer.Progress = TaskObserver.TaskIndeterminate;
			}

			lock (_runningObservers)
			{
				_runningObservers.Add(observer);
			}

			return observer;
		}

		private void OnTaskCreated(ITaskObserver task, object sender = null)
		{
			TaskCreated?.Invoke(sender ?? this, new TaskModifiedEventArgs(task, TaskObserveStatus.Running));
		}

		private void OnTaskCanceled(ITaskObserver task, object sender = null)
		{
			TaskCanceled?.Invoke(sender ?? this, new TaskModifiedEventArgs(task, TaskObserveStatus.Canceled));
		}

		private void OnTaskEnded(ITaskObserver task, object sender = null)
		{
			TaskEnded?.Invoke(sender ?? this, new TaskModifiedEventArgs(task, TaskObserveStatus.Ended));
		}

		public void CancelTask(ITaskObserver task)
		{
			if (task.Status != TaskObserveStatus.Running)
			{
				//nothing to do here, task is not even running
				return;
			}

			OnTaskCanceled(task);

			task.Status = TaskObserveStatus.Canceled;

			lock (_runningObservers)
			{
				_runningObservers.Remove(task);
			}
		}

		public void EndTask(ITaskObserver task)
		{
			if (task.Status != TaskObserveStatus.Running)
			{
				//nothing to do here, task is not even running
				return;
			}

			OnTaskEnded(task);

			task.Status = TaskObserveStatus.Ended;

			lock (_runningObservers)
			{
				_runningObservers.Remove(task);
			}
		}

		public ICollection<ITaskObserver> GetTasks()
		{
			return _runningObservers;
		}


		public IEnumerable<ITaskObserver> GetTasks(ITaskType taskType)
		{
			lock (_runningObservers)
			{
				return _runningObservers.Where(observer => observer.Type == taskType && observer.Exposed);
			}
		}
	}

	/// <summary>
	/// A task manager, responsible for managing task observers.
	/// </summary>
	public interface ITaskManager
	{
		/// <summary>
		/// Begin a task with a certain task type and optional description.
		/// </summary>
		/// <param name="taskType">The task type.</param>
		/// <param name="taskDescription">The task description.</param>
		/// <param name="exposed">Indicate whether the task should be exposed to external search requests.</param>
		/// <param name="indeterminate">Indicate whether the task is indeterminate (unknown total workload).</param>
		/// <returns></returns>
		ITaskObserver BeginTask(ITaskType taskType, string taskDescription = null, bool exposed = true, bool indeterminate = true);

		/// <summary>
		/// Cancel a certain task.
		/// </summary>
		/// <param name="task">The task to cancel.</param>
		void CancelTask(ITaskObserver task);

		/// <summary>
		/// End a certain task.
		/// </summary>
		/// <param name="task">The task to end.</param>
		void EndTask(ITaskObserver task);

		/// <summary>
		/// Get all running tasks with a certain task type. 
		/// </summary>
		/// <param name="taskType">The task type to check for.</param>
		/// <returns>All running tasks that match the given task type.</returns>
		IEnumerable<ITaskObserver> GetTasks(ITaskType taskType);

		/// <summary>
		/// Get all running tasks.
		/// </summary>
		/// <returns>All currently running tasks.</returns>
		ICollection<ITaskObserver> GetTasks();

		/// <summary>
		/// This event will be raised when a newly created <see cref="ITaskObserver"/> is added.
		/// </summary>
		event EventHandler<TaskModifiedEventArgs> TaskCreated;

		/// <summary>
		/// This event will be raised when a <see cref="ITaskObserver"/> is canceled. 
		/// </summary>
		event EventHandler<TaskModifiedEventArgs> TaskCanceled;

		/// <summary>
		/// This event will be raised when a <see cref="ITaskObserver"/> is ended. 
		/// </summary>
		event EventHandler<TaskModifiedEventArgs> TaskEnded;
	}

	/// <summary>
	/// A task type, indicating what a given task does.
	/// </summary>
	public interface ITaskType
	{
		/// <summary>
		/// The actual type of this task (e.g. Download, Save).
		/// </summary>
		string Type { get; set; }

		/// <summary>
		/// The expressed type of this task, as it should be output (e.g. Downloading, Saving).
		/// </summary>
		string ExpressedType { get; set; }
	}

	/// <summary>
	/// A default implementation of the <see cref="ITaskType"/> interface. 
	/// Represents a task type, indicating what a given task does. 
	/// </summary>
	public class TaskType : ITaskType
	{
		public static readonly ITaskType Download = new TaskType("Download", "Downloading");
		public static readonly ITaskType Load = new TaskType("Load", "Loading");
		public static readonly ITaskType Save = new TaskType("Save", "Saving");
		public static readonly ITaskType Unpack = new TaskType("Unpack", "Unpacking");
		public static readonly ITaskType Extract = new TaskType("Extract", "Extracting");
		public static readonly ITaskType Preprocess = new TaskType("Preprocess", "Preprocessing");
		public static readonly ITaskType Prepare = new TaskType("Prepare", "Preparing");
		public static readonly ITaskType Train = new TaskType("Train", "Training");

		public string ExpressedType { get; set; }
		public string Type { get; set; }

		public TaskType(string type, string expressedType)
		{
			Type = type;
			ExpressedType = expressedType;
		}
	}
}
