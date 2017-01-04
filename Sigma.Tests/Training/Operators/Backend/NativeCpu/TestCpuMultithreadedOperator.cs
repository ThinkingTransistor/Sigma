using System;
using System.Threading;
using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Training.Operators;
using Sigma.Core.Training.Operators.Backends.NativeCpu;

namespace Sigma.Tests.Training.Operators.Backend.NativeCpu
{
	public class TestCpuMultithreadedOperator
	{
		private static CpuMultithreadedOperator CreateOperator()
		{
			SigmaEnvironment.Clear();

			return new CpuMultithreadedOperator(new CpuFloat32Handler(), 3, ThreadPriority.Normal);
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorCreate()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();

			Assert.AreEqual(3, cpuOperator.WorkerCount);
			Assert.AreEqual(ExecutionState.None, cpuOperator.State);
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorStart()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();

			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			Assert.AreEqual(ExecutionState.Running, cpuOperator.State);

			Assert.Throws<InvalidOperationException>(() => cpuOperator.Start());

			cpuOperator.SignalStop();

			cpuOperator.WaitForStateChanged();

			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalPause();

			cpuOperator.WaitForStateChanged();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.Start());

			cpuOperator.SignalStop();
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalPause()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalPause();

			cpuOperator.WaitForStateChanged();

			Assert.AreEqual(ExecutionState.Paused, cpuOperator.State);
			
			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalPause());

			cpuOperator.SignalStop();

			cpuOperator.WaitForStateChanged();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalPause());
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalResume()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalResume());

			cpuOperator.SignalPause();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalResume();

			cpuOperator.WaitForStateChanged();

			Assert.AreEqual(ExecutionState.Running, cpuOperator.State);
			
			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalResume());

			cpuOperator.SignalStop();
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalStop()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalStop();

			cpuOperator.WaitForStateChanged();

			Assert.AreEqual(ExecutionState.Stopped, cpuOperator.State);
			
			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalStop());

			cpuOperator.Start();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalPause();

			cpuOperator.WaitForStateChanged();

			cpuOperator.SignalStop();
		}
	}
}
