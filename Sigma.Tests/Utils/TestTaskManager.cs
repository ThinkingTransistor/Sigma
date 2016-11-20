/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;
using System.Collections.Generic;
using System.Linq;

namespace Sigma.Tests.Utils
{
	public class TestTaskManager
	{
		[TestCase]
		public void TestTaskManagerBegin()
		{
			TaskManager manager = new TaskManager();

			ITaskObserver task = manager.BeginTask(TaskType.Preprocess, "mnistdataset", exposed: false, indeterminate: true);

			Assert.AreEqual(TaskType.Preprocess, task.Type);
			Assert.AreEqual("mnistdataset", task.Description);
			Assert.IsFalse(task.Exposed);
			Assert.Less(task.Progress, 0.0f);
		}

		[TestCase]
		public void TestTaskManagerEnd()
		{
			TaskManager manager = new TaskManager();

			ITaskObserver task = manager.BeginTask(TaskType.Preprocess, "mnistdataset", exposed: true, indeterminate: true);

			Assert.IsTrue(manager.GetTasks().Contains(task));

			manager.EndTask(task);

			Assert.IsFalse(manager.GetTasks().Contains(task));
			Assert.AreEqual(TaskObserveStatus.Ended, task.Status);
		}

		[TestCase]
		public void TestTaskManagerCancel()
		{
			TaskManager manager = new TaskManager();

			ITaskObserver task = manager.BeginTask(TaskType.Preprocess, "mnistdataset", exposed: true, indeterminate: true);

			Assert.IsTrue(manager.GetTasks().Contains(task));

			manager.CancelTask(task);

			Assert.IsFalse(manager.GetTasks().Contains(task));
			Assert.AreEqual(TaskObserveStatus.Canceled, task.Status);
		}

		[TestCase]
		public void TestTaskManagerGetTasks()
		{
			TaskManager manager = new TaskManager();

			ITaskObserver task1 = manager.BeginTask(TaskType.Download, "mnistdataset", exposed: true);
			ITaskObserver task2 = manager.BeginTask(TaskType.Download, "mnistdataset", exposed: false);
			ITaskObserver task3 = manager.BeginTask(TaskType.Download, "mnistdataset", exposed: true);

			IEnumerable<ITaskObserver> tasks = manager.GetTasks(TaskType.Download);

			Assert.IsTrue(tasks.Contains(task1));
			Assert.IsFalse(tasks.Contains(task2));
			Assert.IsTrue(tasks.Contains(task3));
		}
	}
}
