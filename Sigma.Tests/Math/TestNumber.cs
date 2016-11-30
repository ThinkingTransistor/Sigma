using System;
using NUnit.Framework;
using Sigma.Core.MathAbstract.Backends.NativeCpu;

namespace Sigma.Tests.Math
{
	public class TestNumber
	{
		[TestCase]
		public void TestNumberGetSet()
		{
			Number<float> number = new Number<float>(0.3f);

			Assert.AreEqual(0.3f, number.Value);

			number.Value = 0.2f;

			Assert.AreEqual(0.2f, number.Value);

			Assert.That(() => number.Value = null, Throws.Exception);
		}
	}
}
