using System;
using ManagedCuda;
using ManagedCuda.CudaBlas;
using NUnit.Framework;
using System.Runtime.InteropServices;

namespace Sigma.Tests
{
	public class TestCUDAInstallation
	{
		static bool cudaInstalled;
		static bool checkedCudaInstalled;

		public static void AssertIgnoreIfCudaUnavailable()
		{
			if (!checkedCudaInstalled)
			{
				try
				{
					new CudaContext();

					cudaInstalled = true;
				}
				catch
				{
					cudaInstalled = false;
				}

				checkedCudaInstalled = true;
			}

			if (!cudaInstalled)
			{
				Assert.Ignore();
			}
		}

		[TestCase]
		public void TestCreateDefaultCUDAContext()
		{
			AssertIgnoreIfCudaUnavailable();

			CudaContext context = new CudaContext();
		}

		[TestCase]
		public void TestCreateCudaBlas()
		{
			AssertIgnoreIfCudaUnavailable();

			CudaBlas cublas = new CudaBlas();
		}
	}
}
