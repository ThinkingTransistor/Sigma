/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Utils;

namespace Sigma.Tests.Utils
{
	public class TestArrayUtils
	{
		[TestCase]
		public void TestArrayUtilsProduct()
		{
			Assert.AreEqual(120L, ArrayUtils.Product(2, 3, 4, 5));
			Assert.AreEqual(-1800L, ArrayUtils.Product(2, 3, 4, 5, -15));
		}

		[TestCase]
		public void TestArrayUtilsRange()
		{
			Assert.AreEqual(new[] { 2, 3, 4, 5 }, ArrayUtils.Range(2, 5, 1));
			Assert.AreEqual(new[] { 10, 9, 8, 7 }, ArrayUtils.Range(10, 7, 1));

			Assert.AreEqual(new[] { 2, 4 }, ArrayUtils.Range(2, 5, 2));
			Assert.AreEqual(new[] { 5, 3 }, ArrayUtils.Range(5, 2, 2));

			Assert.Throws<System.ArgumentException>(() => ArrayUtils.Range(1, 2, -1));
		}

		[TestCase]
		public void TestArrayUtilsPermuteArray()
		{
			Assert.AreEqual(new long[] { 4, 2, 1, 3 }, ArrayUtils.PermuteArray(new long[] { 1, 2, 3, 4 }, new[] { 3, 1, 0, 2 }));
		}

		[TestCase]
		public void TestArrayUtilsGetFlatColumnMappings()
		{
			Assert.AreEqual(new[] { 1, 2, 3, 4, 5, 9 }, ArrayUtils.GetFlatColumnMappings(new[] { new[] { 1, 5 }, new[] { 9 } }));
		}
	}
}
