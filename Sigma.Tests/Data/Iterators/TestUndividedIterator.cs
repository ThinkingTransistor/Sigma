/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Sigma.Core;
using Sigma.Core.Data.Datasets;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Iterators;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;
using Sigma.Core.Utils;

namespace Sigma.Tests.Data.Iterators
{
	public class TestUndividedIterator : BaseLocaleTest
	{
		private static void CreateCsvTempFile(string name)
		{
			File.Create(Path.GetTempPath() + name).Dispose();
			File.WriteAllLines(Path.GetTempPath() + name, new[] { "5.1,3.5,1.4,0.2,Iris-setosa", "4.9,3.0,1.4,0.2,Iris-setosa", "4.7,3.2,1.3,0.2,Iris-setosa" });
		}

		private static void DeleteTempFile(string name)
		{
			File.Delete(Path.GetTempPath() + name);
		}

		[TestCase]
		public void TestUndividedIteratorCreate()
		{
			string filename = ".unittestfile" + nameof(TestUndividedIteratorCreate);

			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new[] { new[] { 0 } } }));
			ExtractedDataset dataset = new ExtractedDataset("test", 1, new DiskCacheProvider(Path.GetTempPath() + "/" + nameof(TestUndividedIteratorCreate)), true, extractor);

			Assert.Throws<ArgumentNullException>(() => new UndividedIterator(null));

			dataset.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestUndividedIteratorYield()
		{
			string filename = ".unittestfile" + nameof(TestUndividedIteratorCreate);

			CreateCsvTempFile(filename);

			SigmaEnvironment.Clear();

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new[] { new[] { 0 } } }));
			ExtractedDataset dataset = new ExtractedDataset("test", 2, new DiskCacheProvider(Path.GetTempPath() + "/" + nameof(TestUndividedIteratorCreate)), true, extractor);
			UndividedIterator iterator = new UndividedIterator(dataset);
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");
			IComputationHandler handler = new CpuFloat32Handler();

			int index = 0;
			foreach (var block in iterator.Yield(handler, sigma))
			{
				if (index == 0)
				{
					Assert.AreEqual(new float[] { 5.1f, 4.9f }, block["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 2));
				}
				else if (index == 1)
				{
					Assert.AreEqual(new float[] { 4.7f }, block["inputs"].GetDataAs<float>().GetValuesArrayAs<float>(0, 1));
				}
				else
				{
					Assert.Fail("There can be a maximum of two iterations, but this is yield iteration 3 (index 2).");
				}

				index++;
			}

			dataset.Dispose();

			DeleteTempFile(filename);
		}
	}
}
