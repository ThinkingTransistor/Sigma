/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using ManagedCuda;
using ManagedCuda.CudaBlas;
using NUnit.Framework;

namespace Sigma.Tests
{
	public class TestCudaInstallation
	{
		private static bool _cudaInstalled;
		private static bool _checkedCudaInstalled;

		public static void AssertIgnoreIfCudaUnavailable()
		{
			if (!_checkedCudaInstalled)
			{
				try
				{
					new CudaContext();

					_cudaInstalled = true;
				}
				catch
				{
					_cudaInstalled = false;
				}

				_checkedCudaInstalled = true;
			}

			if (!_cudaInstalled)
			{
				Assert.Ignore("CUDA installation not found or not working. As CUDA is optional, this test will be ignored.");
			}
		}

		[TestCase]
		public void TestCudaInstallationCreateContext()
		{
			AssertIgnoreIfCudaUnavailable();

			CudaContext context = new CudaContext();
		}

		[TestCase]
		public void TestCudaInstallationCreateCudaBlas()
		{
			AssertIgnoreIfCudaUnavailable();

			CudaBlas cublas = new CudaBlas();
		}
	}
}
