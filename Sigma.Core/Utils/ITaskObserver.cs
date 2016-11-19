/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;

namespace Sigma.Core.Utils
{
	/// <summary>
	/// A task observer, a collection of information about a certain task (does not have to be an actual System.Threading.Task). 
	/// </summary>
	public interface ITaskObserver : IProgress<float>
	{
		/// <summary>
		/// Indicate whether to expose this task to external parts (whether the task can be found by searching or not). 
		/// </summary>
		bool Exposed { get; set; }

		/// <summary>
		/// The task type, i.e. what kind of task this is (e.g. Preprocess, Download - <see cref="TaskManager"/>). 
		/// </summary>
		ITaskType Type { get; set; }

		/// <summary>
		/// The current status of this task (running, ended, canceled - <see cref="TaskObserveStatus"/>).
		/// </summary>
		TaskObserveStatus Status { get; set; }

		/// <summary>
		/// An optional task description, describing further details about this task (e.g. what file is being downloaded). 
		/// </summary>
		string Description { get; set; }

		/// <summary>
		/// The current progress of this task. 
		/// If this is a determinate task, set to the actual progress value, otherwise to a negative value.
		/// </summary>
		float Progress { get; set; }

		/// <summary>
		/// The task time, e.g. when <see cref="ITaskManager.BeginTask(ITaskType,string,bool,bool)"/> was called for this task.
		/// </summary>
		DateTime StartTime { get; set; }

		/// <summary>
		/// The time span since this task was started.
		/// </summary>
		TimeSpan TimeSinceStarted { get; }
	}

	/// <summary>
	/// A default implementation of the <see cref="ITaskObserver"/> interface.
	/// Represents an informative wrapper around any task (does not have to be a System.Threading.Tasks task).
	/// </summary>
	public class TaskObserver : ITaskObserver
	{
		public const float TaskIndeterminate = -1.0f;

		public string Description { get; set; }
		public bool Exposed { get; set; }
		public float Progress { get; set; }
		public TaskObserveStatus Status { get; set; }
		public ITaskType Type { get; set; }
		public DateTime StartTime { get; set; }
		public TimeSpan TimeSinceStarted => DateTime.Now - StartTime;

		public TaskObserver(ITaskType type, string description = null, bool exposed = true)
		{
			if (type == null)
			{
				throw new ArgumentNullException(nameof(type));
			}

			Type = type;
			Description = description;
			Exposed = exposed;
		}

		public void Report(float value)
		{
			float newProgress = value / 100.0f;

			if (newProgress == Progress)
			{
				return;
			}

			Progress = value / 100.0f;

			//Console.WriteLine($"{Type.ExpressedType} {Description} ({Progress * 100:00.0}%)");
		}
	}

	/// <summary>
	/// A task status, used by task manager and task observers to indicate what a given task is up to.
	/// </summary>
	public enum TaskObserveStatus
	{
		Running,
		Ended,
		Canceled
	}
}
