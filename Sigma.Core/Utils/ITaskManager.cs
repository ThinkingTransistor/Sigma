/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A default implementation of the <see cref="ITaskManager"/> interface. 
	/// Represents a task manager, that manages a set of task observers. 
	/// </summary>
	public class TaskManager : ITaskManager
	{
		private IList<ITaskObserver> runningObservers = new List<ITaskObserver>();

		public ITaskObserver BeginTask(ITaskType taskType, string taskDescription = null, bool exposed = true, bool indeterminate = false)
		{
			if (taskType == null)
			{
				throw new ArgumentNullException("Task type cannot be null.");
			}

			ITaskObserver observer = new TaskObserver(taskType, taskDescription, exposed);

			observer.Status = TaskStatus.RUNNING;
			observer.StartTime = DateTime.Now;

			if (indeterminate)
			{
				observer.Progress = TaskObserver.TASK_INDETERMINATE;
			}

			lock (runningObservers)
			{
				runningObservers.Add(observer);
			}

			return observer;	
		}

		public void CancelTask(ITaskObserver task)
		{
			if (task.Status != TaskStatus.RUNNING)
			{
				//nothing to do here, task is not even running
				return;
			}

			task.Status = TaskStatus.CANCELED;

			lock (runningObservers)
			{
				runningObservers.Remove(task);
			}
		}

		public void EndTask(ITaskObserver task)
		{
			if (task.Status != TaskStatus.RUNNING)
			{
				//nothing to do here, task is not even running
				return;
			}

			task.Status = TaskStatus.ENDED;

			lock (runningObservers)
			{
				runningObservers.Remove(task);
			}
		}

		public IEnumerable<ITaskObserver> GetTasks()
		{
			return runningObservers;
		}

		public IEnumerable<ITaskObserver> GetTasks(ITaskType taskType)
		{
			return runningObservers.Where(observer => observer.Type == taskType);
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
		ITaskObserver BeginTask(ITaskType taskType, string taskDescription = null, bool exposed = true, bool indeterminate = false);

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
		IEnumerable<ITaskObserver> GetTasks();
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
		public static readonly ITaskType DOWNLOAD = new TaskType("Download", "Downloading");
		public static readonly ITaskType LOAD = new TaskType("Load", "Loading");
		public static readonly ITaskType SAVE = new TaskType("Save", "Saving");
		public static readonly ITaskType UNPACK = new TaskType("Unpack", "Extracting");
		public static readonly ITaskType EXTRACT = new TaskType("Extract", "Extracting");
		public static readonly ITaskType PREPROCESS = new TaskType("Preprocess", "Preprocessing");
		public static readonly ITaskType PREPARE = new TaskType("Prepare", "Preparing");
		public static readonly ITaskType TRAIN = new TaskType("Train", "Training");

		public string ExpressedType { get; set; }
		public string Type { get; set; }

		public TaskType(string type, string expressedType)
		{
			this.Type = type;
			this.ExpressedType = expressedType;
		}
	}
}
