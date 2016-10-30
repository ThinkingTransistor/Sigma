using NUnit.Framework;
using Sigma.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
