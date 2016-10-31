using System;
using NUnit.Framework;
using Sigma.Core.Utils;

namespace Sigma.Tests.Utils
{
	public class TestRegistry
	{
		[TestCase]
		public void TestRegistryReadWrite()
		{
			Registry registry = new Registry();

			registry["testkey1"] = "testvalue1";
			registry["testkey2"] = "testvalue2";

			Assert.AreEqual(registry["testkey1"], "testvalue1");
			Assert.AreEqual(registry["testkey2"], "testvalue2");
		}
	}
}
