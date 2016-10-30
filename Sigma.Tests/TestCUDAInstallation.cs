using System;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using NUnit.Framework;

namespace Sigma.Tests
{
	public class TestCUDAInstallation
	{
		[TestCase]
		public void TestCreateDefaultCUDAContext()
		{
			CudaContext context = new CudaContext();
		}

		[TestCase]
		public void TestCreateCudaBlas()
		{
			CudaBlas cublas = new CudaBlas();
		}
	}
}
