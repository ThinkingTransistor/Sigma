using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Operators.Backends.NativeCpu;
using Sigma.Core.Training.Operators.Backends.NativeCpu.Workers;
using System;
using System.IO;
using System.Threading;

namespace Sigma.Tests.Training.Operators.Backend.NativeCpu.Workers
{
	public class TestCpuWorker
	{
		private static CpuWorker CreateCpuWorker()
		{
			var @operator = CreateOperator();
			var worker = new CpuWorker(@operator);
			worker.LocalNetwork = @operator.Network;
			worker.LocalTrainingDataIterator = @operator.Trainer.TrainingDataIterator;
			worker.LocalOptimiser = @operator.Trainer.Optimiser;

			return worker;
		}

		private static void RedirectGlobalsToTempPath()
		{
			SigmaEnvironment.Globals["workspace_path"] = Path.GetTempPath();
			SigmaEnvironment.Globals["cache_path"] = Path.GetTempPath() + "sigmacache";
			SigmaEnvironment.Globals["datasets_path"] = Path.GetTempPath() + "sigmadatasets";
		}

		private static CpuMultithreadedOperator CreateOperator()
		{
			SigmaEnvironment.Clear();
			RedirectGlobalsToTempPath();

			CpuMultithreadedOperator @operator = new CpuMultithreadedOperator(new CpuFloat32Handler(), 3, ThreadPriority.Normal);
			@operator.Trainer = new MockTrainer();
			@operator.Trainer.Initialise(@operator.Handler);
			@operator.Network = @operator.Trainer.Network;
			@operator.Sigma = SigmaEnvironment.GetOrCreate("testificate-operatorcreate");

			return @operator;
		}

		[TestCase]
		public void TestCpuWorkerCreate()
		{
			IOperator @operator = new CpuMultithreadedOperator(10);
			CpuWorker worker = new CpuWorker(@operator, @operator.Handler, ThreadPriority.Normal);

			Assert.AreSame(@operator, worker.Operator);
			Assert.AreEqual(worker.ThreadPriority, ThreadPriority.Normal);
			Assert.AreSame(worker.Handler, @operator.Handler);

			worker = new CpuWorker(@operator);

			Assert.AreSame(worker.Handler, @operator.Handler);

			Assert.AreEqual(worker.State, ExecutionState.None);
		}

		[TestCase]
		public void TestCpuWorkerStart()
		{
			CpuWorker worker = CreateCpuWorker();

			worker.Start();
			Assert.AreEqual(worker.State, ExecutionState.Running);

			worker.Start();
			Assert.AreEqual(worker.State, ExecutionState.Running);

			worker.SignalStop();
			Assert.AreNotEqual(worker.State, ExecutionState.Running);

			worker.Start();
			Assert.AreEqual(worker.State, ExecutionState.Running);

			worker.SignalPause();
			Assert.AreNotEqual(worker.State, ExecutionState.Running);

			Assert.Throws<InvalidOperationException>(() => worker.Start());
			Assert.AreNotEqual(worker.State, ExecutionState.Running);

			worker.SignalStop();
		}

		[TestCase]
		public void TestCpuWorkerSignalPause()
		{
			CpuWorker worker = CreateCpuWorker();
			worker.Start();

			worker.SignalPause();
			Assert.AreEqual(worker.State, ExecutionState.Paused);

			worker.SignalPause();
			Assert.AreEqual(worker.State, ExecutionState.Paused);

			worker.SignalStop();
			Assert.AreNotEqual(worker.State, ExecutionState.Paused);

			Assert.Throws<InvalidOperationException>(() => worker.SignalPause());
			Assert.AreNotEqual(worker.State, ExecutionState.Paused);
		}


		[TestCase]
		public void TestCpuWorkerSignalResume()
		{
			CpuWorker worker = CreateCpuWorker();
			worker.Start();

			worker.SignalPause();
			worker.SignalResume();
			Assert.AreEqual(worker.State, ExecutionState.Running);

			worker.SignalResume();
			Assert.AreEqual(worker.State, ExecutionState.Running);

			worker.SignalStop();
			Assert.Throws<InvalidOperationException>(() => worker.SignalResume());

			worker.SignalStop();
		}

		[TestCase]
		public void TestCpuWorkerSignalStop()
		{
			CpuWorker worker = CreateCpuWorker();
			worker.Start();

			worker.SignalStop();
			Assert.AreEqual(worker.State, ExecutionState.Stopped);

			worker.SignalStop();
			Assert.AreEqual(worker.State, ExecutionState.Stopped);

			worker.Start();
			Assert.AreNotEqual(worker.State, ExecutionState.Stopped);

			worker.SignalPause();
			worker.SignalStop();
			Assert.AreEqual(worker.State, ExecutionState.Stopped);

			worker.Start();
			worker.SignalPause();
			worker.SignalResume();
			worker.SignalStop();
			Assert.AreEqual(worker.State, ExecutionState.Stopped);
		}
	}
}