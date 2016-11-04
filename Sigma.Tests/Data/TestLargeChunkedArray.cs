/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using NUnit.Framework;
using Sigma.Core.Data;

namespace Sigma.Tests.Data
{
	public class TestLargeChunkedArray
	{
		[TestCase]
		public void TestLargeChunkedArrayCreateSmall()
		{
			LargeChunkedArray<float> smallArray = new LargeChunkedArray<float>(50000);

			Assert.AreEqual(smallArray.Length, 50000);

			smallArray[0] = 1.0f;
			smallArray[1] = 2.0f;
			smallArray[25000] = 3.0f;
			smallArray.SetValue(4.0f, 49999);

			Assert.AreEqual(smallArray[0], 1.0f);
			Assert.AreEqual(smallArray[1], 2.0f);
			Assert.AreEqual(smallArray.GetValue(25000), 3.0f);
			Assert.AreEqual(smallArray[49999], 4.0f);

			Assert.Throws<IndexOutOfRangeException>(() => smallArray[smallArray.Length] = 0.0f);
		}

		[TestCase]
		public void TestLargeChunkedArrayCreateLarge()
		{
			LargeChunkedArray<float> largeArray = new LargeChunkedArray<float>(5000000);

			Assert.AreEqual(largeArray.Length, 5000000);

			largeArray[0] = 1.0f;
			largeArray[1] = 2.0f;
			largeArray[25000] = 3.0f;
			largeArray.SetValue(6.0f, 2000000);
			largeArray.SetValue(7.0f, 2500000);
			largeArray[largeArray.Length - 1] = 9.0f;

			Assert.AreEqual(largeArray[0], 1.0f);
			Assert.AreEqual(largeArray[1], 2.0f);
			Assert.AreEqual(largeArray.GetValue(25000), 3.0f);
			Assert.AreEqual(largeArray[2000000], 6.0f);
			Assert.AreEqual(largeArray[2500000], 7.0f);
			Assert.AreEqual(largeArray[largeArray.Length - 1], 9.0f);

			Assert.Throws<IndexOutOfRangeException>(() => largeArray[largeArray.Length] = 0.0f);
		}
	}
}
