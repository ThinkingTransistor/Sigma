using System;
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
			SigmaEnvironment environment = SigmaEnvironment.Create("Test");

			return new CpuMultithreadedOperator(environment, new CpuFloat32Handler(), 10);
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorCreate()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();

			Assert.AreEqual(cpuOperator.WorkerCount, 10);
			Assert.AreEqual(cpuOperator.State, ExecutionState.None);
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorStart()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();

			cpuOperator.Start();

			Assert.AreEqual(cpuOperator.State, ExecutionState.Running);

			Assert.Throws<InvalidOperationException>(() => cpuOperator.Start());

			cpuOperator.SignalStop();

			cpuOperator.Start();

			cpuOperator.SignalPause();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.Start());

			cpuOperator.SignalStop();
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalPause()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			cpuOperator.SignalPause();

			Assert.AreEqual(cpuOperator.State, ExecutionState.Paused);

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalPause());

			cpuOperator.SignalStop();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalPause());
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalResume()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalResume());

			cpuOperator.SignalPause();

			cpuOperator.SignalResume();

			Assert.AreEqual(cpuOperator.State, ExecutionState.Running);

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalResume());

			cpuOperator.SignalStop();
		}

		[TestCase]
		public void TestCpuMultithreadedOperatorSignalStop()
		{
			CpuMultithreadedOperator cpuOperator = CreateOperator();
			cpuOperator.Start();

			cpuOperator.SignalStop();

			Assert.AreEqual(cpuOperator.State, ExecutionState.Stopped);

			Assert.Throws<InvalidOperationException>(() => cpuOperator.SignalStop());

			cpuOperator.Start();

			cpuOperator.SignalPause();

			cpuOperator.SignalStop();
		}
	}
}
