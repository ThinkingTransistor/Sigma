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
			DataBuffer<double> buffer = new DataBuffer<double>(length);

			Assert.AreEqual(buffer.Length, length);
		}
	}
}
