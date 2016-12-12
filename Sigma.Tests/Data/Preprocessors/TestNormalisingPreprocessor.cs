/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends;
using Sigma.Core.MathAbstract;
using System;
using System.Collections.Generic;
using System.Linq;
using Sigma.Core.Handlers.Backends.DiffSharp.NativeCpu;
using Sigma.Core.MathAbstract.Backends.DiffSharp.NativeCpu;

namespace Sigma.Tests.Data.Preprocessors
{
	public class TestNormalisingPreprocessor
	{
		private static Dictionary<string, INDArray> GetNamedArrayTestData()
		{
			return new Dictionary<string, INDArray>() { ["test"] = new NDArray<float>(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 1, 1, 9) };
		}

		[TestCase]
		public void TestNormalisingPreprocessorCreate()
		{
			Assert.Throws<ArgumentException>(() => new NormalisingPreprocessor(2, 1, "test"));
			Assert.Throws<ArgumentException>(() => new NormalisingPreprocessor(1, 2, 2, 1));

			NormalisingPreprocessor normaliser = new NormalisingPreprocessor(1, 3, 0, 1, "test");

			Assert.AreEqual(1, normaliser.MinInputValue);
			Assert.AreEqual(3, normaliser.MaxInputValue);
			Assert.AreEqual(0, normaliser.MinOutputValue);
			Assert.AreEqual(1, normaliser.MaxOutputValue);
			Assert.AreEqual(new[] { "test" }, normaliser.ProcessedSectionNames);
		}

		[TestCase]
		public void TestNormalisingPreprocessorExtractDirect()
		{
			NormalisingPreprocessor normaliser = new NormalisingPreprocessor(1, 9, 0, 1, "test");
			IComputationHandler handler = new CpuFloat32Handler();

			Dictionary<string, INDArray> extracted = normaliser.ExtractDirectFrom(GetNamedArrayTestData(), 1, handler);

			Assert.AreEqual(new[] { 0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.0f }, extracted["test"].GetDataAs<float>().GetValuesArrayAs<float>(0, 9).ToArray());
		}
	}
}
