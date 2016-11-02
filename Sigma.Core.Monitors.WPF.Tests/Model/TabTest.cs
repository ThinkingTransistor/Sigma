using System;
using NUnit.Framework;
using Sigma.Core.Monitors.WPF.Model;

namespace Sigma.Core.Monitors.WPF.Tests.Model
{
	public class TabTest
	{
		[TestCase]
		public void TestComparison()
		{
			Tab a1 = "a";
			Tab a2 = new Tab("a");
			Tab b1 = new Tab("b");

			Assert.IsTrue(a1 == a2);
			Assert.IsFalse(a1 == b1);
			Assert.IsFalse(a1 == null);

			Assert.AreEqual(a1, a2);
			Assert.AreNotEqual(a1, b1);
			Assert.AreNotEqual(a1, null);

			Assert.Zero(a1.CompareTo(a2));
			Assert.NotZero(a1.CompareTo(b1));
			Assert.NotZero(a1.CompareTo(null));

			Assert.AreEqual(a1.ToString(), a1.Title);

			Assert.Throws<ArgumentException>(() => a1.Equals(3));
		}
	}
}
