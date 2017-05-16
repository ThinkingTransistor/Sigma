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
	public class TestMinibatchIterator : BaseLocaleTest
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
		public void TestMinibatchIteratorCreate()
		{
			string filename = ".unittestfile" + nameof(TestMinibatchIteratorCreate);

			CreateCsvTempFile(filename);
			SigmaEnvironment.Clear();

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new[] { new[] { 0 } } }));
			ExtractedDataset dataset = new ExtractedDataset("test", 1, new DiskCacheProvider(Path.GetTempPath() + "/" + nameof(TestMinibatchIteratorYield)), true, extractor);

			Assert.Throws<ArgumentException>(() => new MinibatchIterator(-3, dataset));
			Assert.Throws<ArgumentNullException>(() => new MinibatchIterator(1, null));

			dataset.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestMinibatchIteratorYieldAligned()
		{
			TestMinibatchIteratorYield(3);
		}

		[TestCase]
		public void TestMinibatchIteratorYieldUnaligned()
		{
			TestMinibatchIteratorYield(1);
		}

		public void TestMinibatchIteratorYield(int minibatchSize)
		{
			string filename = ".unittestfile" + nameof(TestMinibatchIteratorYield);

			CreateCsvTempFile(filename);
			SigmaEnvironment.Clear();

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new[] { new[] { 0 } } }));
			ExtractedDataset dataset = new ExtractedDataset("test", 1, new DiskCacheProvider(Path.GetTempPath() + "/" + nameof(TestMinibatchIteratorYield)), true, extractor);
			MinibatchIterator iterator = new MinibatchIterator(minibatchSize, dataset);
			IComputationHandler handler = new CpuFloat32Handler();
			SigmaEnvironment sigma = SigmaEnvironment.Create("test");

			Assert.Throws<ArgumentNullException>(() => iterator.Yield(null, null).GetEnumerator().MoveNext());
			Assert.Throws<ArgumentNullException>(() => iterator.Yield(handler, null).GetEnumerator().MoveNext());
			Assert.Throws<ArgumentNullException>(() => iterator.Yield(null, sigma).GetEnumerator().MoveNext());

			int index = 0;
			foreach (var block in iterator.Yield(handler, sigma))
			{
				//pass through each more than 5 times to ensure consistency
				if (index++ > 20)
				{
					break;
				}

				Assert.Contains(block["inputs"].GetValue<float>(0, 0, 0), new float[] { 5.1f, 4.9f, 4.7f });
			}

			dataset.Dispose();

			DeleteTempFile(filename);
		}
	}
}
