/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data
{
	public class TestDataBuffer
	{
		[TestCase]
		public void TestDataBufferCreate()
		{
			long length = 400000L;
			DataBuffer<double> rootBuffer = new DataBuffer<double>(length);

			Assert.AreEqual(rootBuffer.Length, length);
			Assert.AreSame(rootBuffer.Type, DataTypes.FLOAT64);

			DataBuffer<double> childBufferL2 = new DataBuffer<double>(rootBuffer, 100L, length - 200L);

			Assert.AreSame(rootBuffer.Type, childBufferL2.Type);
			Assert.AreSame(childBufferL2.GetUnderlyingBuffer(), rootBuffer);
			Assert.AreSame(childBufferL2.GetUnderlyingRootBuffer(), rootBuffer);
			Assert.AreEqual(childBufferL2.Offset, 100L);
			Assert.AreEqual(childBufferL2.Length, length - 200L);

			DataBuffer<double> childBufferL3 = new DataBuffer<double>(childBufferL2, 100L, length - 400L);

			Assert.AreSame(rootBuffer.Type, childBufferL3.Type);
			Assert.AreSame(childBufferL3.GetUnderlyingRootBuffer(), rootBuffer);
			Assert.AreSame(childBufferL3.GetUnderlyingBuffer(), childBufferL2);
			Assert.AreEqual(childBufferL3.Offset, 200L);
			Assert.AreEqual(childBufferL3.Length, length - 400L);

			Assert.Throws<ArgumentException>(() => new DataBuffer<double>(childBufferL3, 0L, length));

			DataBuffer<double> arrayBuffer = new DataBuffer<double>(new LargeChunkedArray<double>(10000), 10L, 9000L);

			Assert.AreSame(arrayBuffer.Type, DataTypes.FLOAT64);
		}

		[TestCase]
		public void TestDataBufferCopy()
		{
			long length = 400000L;
			DataBuffer<double> rootBuffer = new DataBuffer<double>(length);
			DataBuffer<double> childBufferL2 = new DataBuffer<double>(rootBuffer, 100L, length - 200L);

			IDataBuffer<double> childBufferL2Copy = childBufferL2.Copy();

			Assert.AreSame(childBufferL2.Data, childBufferL2Copy.Data);
			Assert.AreSame(childBufferL2.GetUnderlyingBuffer(), childBufferL2Copy.GetUnderlyingBuffer());
			Assert.AreSame(childBufferL2.GetUnderlyingRootBuffer(), childBufferL2Copy.GetUnderlyingRootBuffer());
			Assert.AreEqual(childBufferL2.Length, childBufferL2Copy.Length);
			Assert.AreEqual(childBufferL2.Offset, childBufferL2Copy.Offset);
			Assert.AreEqual(childBufferL2.RelativeOffset, childBufferL2Copy.RelativeOffset);
		}

		[TestCase]
		public void TestDataBufferModifySingle()
		{
			long length = 400000L;
			long offset = 100L;
			DataBuffer<double> rootBuffer = new DataBuffer<double>(length);
			DataBuffer<double> childBufferL2 = new DataBuffer<double>(rootBuffer, offset, length - offset * 2);
			DataBuffer<double> childBufferL3 = new DataBuffer<double>(childBufferL2, offset, length - offset * 4);

			rootBuffer.SetValue(7.0f, 1000);
			childBufferL2.SetValue(8.0f, 901);
			childBufferL3.SetValue(9.0f, 802);

			Assert.AreEqual(7.0f, rootBuffer.GetValue(1000));
			Assert.AreEqual(7.0f, childBufferL2.GetValue(900));
			Assert.AreEqual(7.0f, childBufferL3.GetValue(800));

			Assert.AreEqual(8.0f, rootBuffer.GetValue(1001));
			Assert.AreEqual(9.0f, rootBuffer.GetValue(1002));
		}

		[TestCase]
		public void TestDataBufferModifyMultiple()
		{
			long length = 400000L;
			long offset = 100L;
			DataBuffer<double> rootBuffer = new DataBuffer<double>(length);
			DataBuffer<double> childBufferL2 = new DataBuffer<double>(rootBuffer, offset, length - offset * 2);
			DataBuffer<double> childBufferL3 = new DataBuffer<double>(childBufferL2, offset, length - offset * 4);

			rootBuffer.SetValues(new double[] { 0.0, 1.1, 2.2, 3.3, 4.4, 5.5 }, 1, 200, 4);
			childBufferL2.SetValues(new double[] { 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1 }, 1, 104, 3);
			childBufferL3.SetValues(new double[] { 8.8, 9.9, 10.1, 11.11, 12.12, 13.13 }, 0, 7, 3);

			Assert.AreEqual(new double[] { 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1 }, childBufferL3.GetValuesArray(0, 10).TryGetValuesPackedArray());
			Assert.AreEqual(new int[] { 1, 2, 3, 4, 6, 7, 8, 9, 10, 10 }, rootBuffer.GetValuesArrayAs<int>(200, 10).TryGetValuesPackedArray());
		}
	}
}
