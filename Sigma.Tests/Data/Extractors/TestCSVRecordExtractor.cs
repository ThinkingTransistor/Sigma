/* 
MIT License

Copyright (c) 2016-2017 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using Sigma.Core.Data.Extractors;
using Sigma.Core.Data.Readers;
using Sigma.Core.Data.Sources;
using Sigma.Core.Handlers.Backends.SigmaDiff.NativeCpu;

namespace Sigma.Tests.Data.Extractors
{
	public class TestCsvRecordExtractor
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
		public void TestCsvRecordExtractorCreate()
		{
			string filename = ".unittestfile" + nameof(TestCsvRecordExtractorCreate);

			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, int[][]> { ["inputs"] = new[] { new[] { 0 } } }));

			Assert.AreSame(source, extractor.Reader.Source);
			Assert.AreEqual(new[] { "inputs" }, extractor.SectionNames);

			source.Dispose();

			DeleteTempFile(filename);
		}

		[TestCase]
		public void TestCsvRecordExtractorExtract()
		{
			string filename = ".unittestfile" + nameof(TestCsvRecordExtractorCreate);

			CreateCsvTempFile(filename);

			FileSource source = new FileSource(filename, Path.GetTempPath());
			CsvRecordExtractor extractor = (CsvRecordExtractor) new CsvRecordReader(source).Extractor(new CsvRecordExtractor(new Dictionary<string, IList<int>>() { ["inputs"] = new[] { 0, 1, 2 } }));

			Assert.Throws<InvalidOperationException>(() => extractor.ExtractDirect(3, new CpuFloat32Handler()));

			extractor.Prepare();

			Assert.Throws<ArgumentNullException>(() => extractor.ExtractDirect(3, null));
			Assert.Throws<ArgumentException>(() => extractor.ExtractDirect(-1, new CpuFloat32Handler()));

			Assert.AreEqual(5, extractor.ExtractDirect(3, new CpuFloat32Handler())["inputs"].GetValue<int>(0, 0, 0));

			source.Dispose();

			DeleteTempFile(filename);
		}
	}
}
