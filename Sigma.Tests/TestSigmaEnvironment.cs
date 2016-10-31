using NUnit.Framework;
using Sigma.Core;

namespace Sigma.Tests
{
	class TestSigmaEnvironment
	{
		[TestCase]
		public void TestSigmaEnvironmentCreate()
		{
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			Assert.AreEqual("test", sigma.Name);
		}
	}
}
