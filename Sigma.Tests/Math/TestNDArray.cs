/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data;
using Sigma.Core.MathAbstract.Backends.DiffSharp;

namespace Sigma.Tests.Math
{
	public class TestNdArray
	{
		[TestCase]
		public void TestNdArrayCreate()
		{
			ADNDArray<int> intArray = new ADNDArray<int>(data: new[] { 1, 2, 3, 4, 5, 6, 7, 8 }, shape: new long[] { 2, 4 });

			Assert.AreEqual(new[] { 1, 2, 3, 4, 5, 6, 7, 8 }, intArray.GetDataAs<int>().GetValuesArrayAs<int>(0, 8));
			Assert.AreEqual(new long[] { 2, 4 }, intArray.Shape);
			Assert.AreEqual(new long[] { 4, 1 }, intArray.Strides);

			ADNDArray<float> floatArray = new ADNDArray<float>(new DataBuffer<float>(new[] { 1.1f, 2.2f, 3.3f, 4.4f }));

			Assert.AreEqual(new[] { 1.1f, 2.2f, 3.3f, 4.4f }, floatArray.GetDataAs<float>().GetValuesArray(0, 4));
			Assert.AreEqual(new long[] { 1, 4 }, floatArray.Shape);
			Assert.AreEqual(new long[] { 4, 1 }, floatArray.Strides);
		}

		[TestCase]
		public void TestNdArrayReshape()
		{
			ADNDArray<int> array = new ADNDArray<int>(data: new[] { 1, 2, 3, 4, 5, 6, 7, 8 }, shape: new long[] { 2, 4 });

			ADNDArray<int> reshaped = (ADNDArray<int>) array.Reshape(4, 2);

			array.ReshapeSelf(4, 2);

			Assert.AreEqual(new long[] { 4, 2 }, array.Shape);
			Assert.AreEqual(new long[] { 4, 2 }, reshaped.Shape);

			Assert.AreEqual(new long[] { 2, 1 }, reshaped.Strides);
			Assert.AreEqual(new long[] { 2, 1 }, array.Strides);
		}

		[TestCase]
		public void TestNdArrayPermute()
		{
			ADNDArray<int> array = new ADNDArray<int>(data: new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, shape: new long[] { 2, 2, 3 });

			ADNDArray<int> permuted = (ADNDArray<int>) array.Permute(2, 1, 0);

			array.PermuteSelf(2, 1, 0);

			Assert.AreEqual(new long[] { 3, 2, 2 }, permuted.Shape);
			Assert.AreEqual(new long[] { 3, 2, 2 }, array.Shape);

			Assert.AreEqual(new long[] { 4, 2, 1 }, permuted.Strides);
			Assert.AreEqual(new long[] { 4, 2, 1 }, array.Strides);
		}

		[TestCase]
		public void TestNdArrayTranspose()
		{
			ADNDArray<int> array = new ADNDArray<int>(data: new[] { 1, 2, 3, 4, 5, 6, 7, 8 }, shape: new long[] { 2, 4 });

			ADNDArray<int> transposed = (ADNDArray<int>) array.Transpose();

			array.TransposeSelf();

			Assert.AreEqual(new long[] { 4, 2 }, transposed.Shape);
			Assert.AreEqual(new long[] { 4, 2 }, array.Shape);

			Assert.AreEqual(new long[] { 2, 1 }, array.Strides);
			Assert.AreEqual(new long[] { 2, 1 }, array.Strides);
		}
	}
}
