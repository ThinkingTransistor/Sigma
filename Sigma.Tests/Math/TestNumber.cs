using System;
using NUnit.Framework;
using Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu;

namespace Sigma.Tests.Math
{
	public class TestNumber
	{
		[TestCase]
		public void TestNumberGetSet()
		{
			ADNumber<float> adNumber = new ADNumber<float>(0.3f);

			Assert.AreEqual(0.3f, adNumber.Value);

			adNumber.Value = 0.2f;

			Assert.AreEqual(0.2f, adNumber.Value);

			Assert.That(() => adNumber.Value = null, Throws.Exception);
		}
	}
}
