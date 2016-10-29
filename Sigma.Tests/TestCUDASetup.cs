using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using ManagedCuda;
using ManagedCuda.CudaBlas;

namespace Sigma.Tests
{
	[TestClass]
	public class TestCUDASetup
	{
		[TestMethod]
		public void TestCreateDefaultCUDAContext()
		{
			CudaContext context = new CudaContext();
		}

		[TestMethod]
		public void TestCreateCudaBlas()
		{
			CudaBlas cublas = new CudaBlas();
		}
	}
}
