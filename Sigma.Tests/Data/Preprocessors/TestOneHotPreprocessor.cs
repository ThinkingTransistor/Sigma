/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using NUnit.Framework;
using Sigma.Core.Data.Preprocessors;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends;
using Sigma.Core.Math;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sigma.Tests.Data.Preprocessors
{
	public class TestOneHotPreprocessor
	{
		private static Dictionary<string, INDArray> GetNamedArrayTestData()
		{
			return new Dictionary<string, INDArray>() { ["test"] = new NDArray<float>(new float[] { 0, 2, 1 }, 3, 1, 1) };
		}

		[TestCase]
		public void TestOneHotPreprocessorCreate()
		{
			Assert.Throws<ArgumentNullException>(() => new OneHotPreprocessor(sectionName: "section"));

			OneHotPreprocessor preprocessor = new OneHotPreprocessor("section", 1, 2, 3, 4);

			Assert.AreEqual(new string[] { "section" }, preprocessor.ProcessedSectionNames);
		}

		[TestCase]
		public void TestOneHotPreprocessorExtract()
		{
			OneHotPreprocessor preprocessor = new OneHotPreprocessor("test", minValue: 0, maxValue: 2);
			IComputationHandler handler = new CPUFloat32Handler();

			Dictionary<string, INDArray> extracted = preprocessor.ExtractDirectFrom(GetNamedArrayTestData(), 2, handler);

			Assert.AreEqual(new float[] { 1, 0, 0, 0, 0, 1 }, extracted["test"].GetDataAs<float>().GetValuesArrayAs<float>(0, 6).ToArray());
		}
	}
}
